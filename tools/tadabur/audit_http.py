"""Shared HTTP scaffolding for the local audit UIs (poison audit + waqf events).

Both `tadabur.audit_ui` (P3.5 poison audit) and `tadabur.waqf_audit_ui` (waqf
event adjudication) serve the same thing: a single-page app plus a JSON API plus
sandboxed clip-audio streaming, all with no database or framework. This module
owns the parts that are identical between them — the ``Content-Type`` sniffing, a
base request handler with JSON/bytes senders and a path-traversal-proof audio
route, and the threading-server builder — so neither UI re-implements (or drifts
on) that boilerplate. Each UI subclasses :class:`AuditHandler`, implements its own
``do_GET`` / ``do_POST`` routing, and reads its state object off ``self.state``.
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


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


class AuditHandler(BaseHTTPRequestHandler):
    """Base request handler: silent logging, JSON/bytes senders, audio streaming.

    A UI-specific subclass is bound to its state object (via :func:`serve`) and
    implements the routing; this base keeps the transport boring and identical
    across UIs so a review of one covers both.
    """

    state: object  # bound onto the subclass by serve()

    def log_message(self, *args: object) -> None:  # noqa: D401 - quiet the default stderr spam
        return

    def send_json(self, obj: object, status: int = 200) -> None:
        body = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def send_bytes(self, data: bytes, content_type: str, status: int = 200) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def serve_audio(self, audio_dir: Path, name: str) -> None:
        """Stream ``name`` from ``audio_dir``, refusing anything that escapes it.

        ``name`` is a sampler ``local_audio_path`` (a flat, separator-free name);
        it is resolved under ``audio_dir`` and rejected if the resolved target is
        not inside it, so a crafted ``../`` cannot read outside the clip store.
        """
        root = audio_dir.resolve()
        target = (root / name).resolve()
        if root not in target.parents or not target.is_file():
            self.send_json({"error": "audio not found"}, status=404)
            return
        data = target.read_bytes()
        self.send_bytes(data, sniff_audio_content_type(data))


def serve(handler_cls: type[AuditHandler], state: object, port: int, host: str = "127.0.0.1") -> ThreadingHTTPServer:
    """Build (but do not block on) the threading HTTP server for ``state``.

    The state is bound onto a per-server subclass of ``handler_cls`` so each
    request-handler instance can reach the loaded worklist and label store without
    globals. ``host`` defaults to loopback; pass ``0.0.0.0`` to expose the UI on
    the LAN so a human on another device can grade the clips.
    """

    class _BoundHandler(handler_cls):  # type: ignore[valid-type, misc]
        pass

    _BoundHandler.state = state
    return ThreadingHTTPServer((host, port), _BoundHandler)

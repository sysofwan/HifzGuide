"""The S6 listening session: one page, the 150 drawn tlog clips, atomic verdicts.

Architect r20 Q4/Q7. The page never learns a clip's stratum, its filename label or its assigned
label: the candidates are anonymous ayah texts in salted order. The server derives the frozen
verdict from the ayah the owner picks (``right ayah`` when it is the label under test, ``other
ayah`` otherwise), and stores the full provenance with every verdict.

Usage, on the box (``scripts/issue-191-ident.sh ui launch`` drives this)::

    python -m tadabur.tlog_audit_ui \\
        --worklist tools/tadabur/tlog/audit/worklist.jsonl \\
        --audio-dir /root/scratch/tlog/s6/audio \\
        --verdicts tools/tadabur/tlog/audit/verdicts.jsonl \\
        --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from training.tashkeel_eval import write_text_atomically

from . import tlog_ident_run as run
from .audit_http import AuditHandler, serve
from .audit_ui import DEFAULT_QURAN_DB, uthmani_index

PAGE_PATH = Path(__file__).parent / "tlog_audit_page.html"


def _sha_key(row: Mapping[str, object]) -> str:
    return run._item_key(row)


def _ayah_counts(quran_db: Path) -> dict[int, int]:
    """Ayat per surah, so "another ayah" is a reference the mushaf actually has."""
    with sqlite3.connect(quran_db) as conn:
        return {int(surah): int(count) for surah, count in
                conn.execute("SELECT surah, COUNT(*) FROM ayahs GROUP BY surah")}


def _is_ayah(reference: str, ayah_counts: Mapping[int, int]) -> bool:
    surah, _, ayah = reference.partition(":")
    if not (surah.isdigit() and ayah.isdigit()):
        return False
    return 1 <= int(surah) <= 114 and 1 <= int(ayah) <= ayah_counts.get(int(surah), 0)


@dataclass
class Session:
    """The worklist, its texts, the audio, the mushaf's ayat and the verdicts on disk."""

    rows: list[dict]
    audio_dir: Path
    verdicts_path: Path
    texts: dict[str, str]
    ayah_counts: dict[int, int] = field(default_factory=dict)
    verdicts: dict[str, dict] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._by_key = {_sha_key(row): row for row in self.rows}
        if len(self._by_key) != len(self.rows):
            raise ValueError("two worklist rows share a page key")

    @classmethod
    def load(cls, worklist: Path, audio_dir: Path, verdicts: Path,
             quran_db: Path = DEFAULT_QURAN_DB) -> "Session":
        rows = [json.loads(line) for line in Path(worklist).read_text(encoding="utf-8")
                .splitlines() if line.strip()]
        labels = {label for row in rows
                  for label in (row.get("filename_label"), row.get("class_label")) if label}
        texts = uthmani_index(Path(quran_db), labels)
        missing = sorted(labels - set(texts))
        if missing:
            raise ValueError(f"{quran_db} has no Uthmani text for {len(missing)} of "
                             f"{len(labels)} ayah labels ({', '.join(missing[:3])}); "
                             "point --quran-db at a checkout's data/quran.db")
        return cls(rows=rows, audio_dir=Path(audio_dir), verdicts_path=Path(verdicts),
                   texts=texts, ayah_counts=_ayah_counts(Path(quran_db)),
                   verdicts=run.read_verdicts(Path(verdicts)))

    def payload(self) -> dict:
        """The served payload, with each item's audio URL and nothing identifying."""
        draw = run.AuditDraw(items=tuple(self.rows), strata={})
        payload = run.audit_payload(draw, texts=self.texts, verdicts=self.verdicts)
        for item in payload["items"]:
            row = self._by_key[item["key"]]
            item["audio_url"] = f"/audio/{row['local_audio']}"
            item["duration_s"] = row["duration_s"]
        return payload

    def record(self, key: str, choice: str, ayah: str | None, note: str) -> dict:
        """Store one answer; the frozen verdict is derived from the label under test."""
        row = self._by_key[key]
        if choice == "candidate":
            if not ayah:
                raise ValueError("a candidate answer needs its ayah")
            verdict = run.RIGHT if ayah == run.label_under_test(row) else run.OTHER
        elif choice == "another":
            if not ayah or not _is_ayah(ayah, self.ayah_counts):
                raise ValueError("another ayah must be a surah:ayah the mushaf has, such as 2:255")
            verdict = run.OTHER
        elif choice in (run.SILENCE, run.NOT_RECITATION, run.UNCLEAR):
            verdict, ayah = choice, None
        else:
            raise ValueError(f"{choice!r} is not a page choice")
        saved = {**{field: row.get(field) for field in (
            "split", "shard_file", "row", "audio_path", "audio_sha256")},
            "verdict": verdict, "ayah": ayah, "note": note}
        verdicts = {**self.verdicts, key: saved}
        write_text_atomically(
            self.verdicts_path,
            "".join(json.dumps(verdicts[item], ensure_ascii=False, sort_keys=True) + "\n"
                    for item in sorted(verdicts)))
        self.verdicts = verdicts
        return saved


class Handler(AuditHandler):
    """Routes: the page, the anonymous payload, clip audio and verdict submission."""

    state: Session

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler's interface
        path = unquote(urlparse(self.path).path)
        if path in ("/", "/index.html"):
            self.send_bytes(PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/items":
            self.send_json(self.state.payload())
        elif path.startswith("/audio/"):
            self.serve_audio(self.state.audio_dir, path[len("/audio/"):])
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
        try:
            saved = self.state.record(str(payload.get("key", "")),
                                      str(payload.get("choice", "")),
                                      payload.get("ayah"), str(payload.get("note", "")))
        except KeyError:
            self.send_json({"error": "unknown item"}, status=404)
            return
        except ValueError as error:
            self.send_json({"error": str(error)}, status=400)
            return
        self.send_json({"progress": self.state.payload()["progress"],
                        "verdict": saved["verdict"]})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--worklist", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--verdicts", type=Path, required=True)
    parser.add_argument("--quran-db", type=Path, default=DEFAULT_QURAN_DB)
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", default="127.0.0.1",
                        help="bind address; 0.0.0.0 to answer from the owner's phone")
    args = parser.parse_args()
    state = Session.load(args.worklist, args.audio_dir, args.verdicts, args.quran_db)
    progress = state.payload()["progress"]
    print(f"{progress['answered']} of {progress['total']} answered. "
          f"Serving on http://{args.host}:{args.port}/", flush=True)
    serve(Handler, state, args.port, args.host).serve_forever()


if __name__ == "__main__":
    main()

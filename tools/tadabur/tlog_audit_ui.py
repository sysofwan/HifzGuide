"""The S6 listening session: one page, the 150 drawn tlog clips, atomic verdicts.

Architect r20 Q4/Q7. The page never learns a clip's stratum, its filename label or its assigned
label: the candidates are anonymous ayah texts. The server derives the frozen verdict from the
ayah the owner picks, comparing **text classes** (identical recitations share a class, so a
repeated ayah is not scored wrong for being cited under its other number), and stores the raw
heard reference beside the canonical one with full provenance.

Every save is serialized under one lock and replaces the answer the page last saw (``previous``);
an answer that changed elsewhere is rejected rather than silently overwritten.

Usage, on the box (``scripts/issue-191-ident.sh ui launch`` drives this)::

    python -m tadabur.tlog_audit_ui \\
        --worklist tools/tadabur/tlog/audit/worklist.jsonl \\
        --audio-dir /root/scratch/tlog/s6/audio \\
        --verdicts /root/scratch/tlog/s6/verdicts.jsonl \\
        --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import argparse
import json
import threading
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import unquote, urlparse

from training.tashkeel_eval import write_text_atomically

from . import tlog_ident_run as run
from .audit_http import AuditHandler, serve
from .audit_ui import DEFAULT_QURAN_DB, uthmani_index
from .tlog_ident import ReferenceDatabase, canonical_label, load_references

PAGE_PATH = Path(__file__).parent / "tlog_audit_page.html"
INVALID_AYAH = "another ayah must be a surah:ayah the mushaf has, such as 2:255"


def _sha_key(row: Mapping[str, object]) -> str:
    return run._item_key(row)


def stored_answer(verdict: Mapping[str, object] | None) -> object:
    """The stored answer as the page names it in ``previous``: none, or ``[verdict, heard]``."""
    if verdict is None:
        return None
    return [verdict.get("verdict"), verdict.get("heard_ayah", verdict.get("ayah"))]


class StaleAnswer(Exception):
    """The answer replaces one the client did not see: the item was answered since."""


@dataclass
class Session:
    """The worklist, its texts, the audio, the reference classes and the verdicts on disk."""

    rows: list[dict]
    audio_dir: Path
    verdicts_path: Path
    texts: dict[str, str]
    database: ReferenceDatabase
    verdicts: dict[str, dict] = field(default_factory=dict)
    _by_key: dict[str, dict] = field(default_factory=dict)
    _under_test: dict[str, str | None] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self) -> None:
        self._by_key = {_sha_key(row): row for row in self.rows}
        if len(self._by_key) != len(self.rows):
            raise ValueError("two worklist rows share a page key")
        self._under_test = {_sha_key(row): self.canonical(run.label_under_test(row))
                            for row in self.rows}

    def canonical(self, label: object) -> str | None:
        """The text class's canonical label: equal labels mean the same recitation."""
        return None if label is None else canonical_label(self.database, label)

    @classmethod
    def load(cls, worklist: Path, audio_dir: Path, verdicts: Path,
             quran_db: Path = DEFAULT_QURAN_DB,
             database: ReferenceDatabase | None = None) -> "Session":
        rows = [json.loads(line) for line in Path(worklist).read_text(encoding="utf-8")
                .splitlines() if line.strip()]
        database = load_references() if database is None else database
        labels = set()
        for row in rows:
            for label in (row.get("filename_label"), row.get("class_label")):
                if not label:
                    continue
                name = canonical_label(database, label)
                if name is None:
                    raise ValueError(f"{label} is not an ayah of the reference database")
                labels.add(name)
        texts = uthmani_index(Path(quran_db), labels)
        missing = sorted(labels - set(texts))
        if missing:
            raise ValueError(f"{quran_db} has no Uthmani text for {len(missing)} of "
                             f"{len(labels)} ayah labels ({', '.join(missing[:3])}); "
                             "point --quran-db at a checkout's data/quran.db")
        return cls(rows=rows, audio_dir=Path(audio_dir), verdicts_path=Path(verdicts),
                   texts=texts, database=database,
                   verdicts=run.read_verdicts(Path(verdicts)))

    def payload(self) -> dict:
        """The served payload, with each item's audio URL and nothing identifying."""
        draw = run.AuditDraw(items=tuple(self.rows), strata={})
        payload = run.audit_payload(draw, texts=self.texts, verdicts=self.verdicts,
                                    canonical=self.canonical)
        for item in payload["items"]:
            row = self._by_key[item["key"]]
            item["audio_url"] = f"/audio/{row['local_audio']}"
            item["duration_s"] = row["duration_s"]
        return payload

    def record(self, key: str, choice: str, ayah: object, note: str,
               previous: object) -> dict:
        """Store one answer, replacing the answer the page last saw (``previous``).

        ``previous`` is :func:`stored_answer` as the page received it: ``None`` for an item it
        saw unanswered, ``[verdict, heard]`` otherwise. A save whose snapshot no longer matches
        raises :class:`StaleAnswer`; nothing changes in that case or on any refusal.
        """
        row = self._by_key[key]
        heard_ayah: str | None = None
        reference: str | None = None
        if choice in ("candidate", "another"):
            if not ayah:
                raise ValueError(INVALID_AYAH if choice == "another"
                                 else "a candidate answer needs its ayah")
            heard_ayah = str(ayah)
            reference = self.canonical(heard_ayah)
            if reference is None:
                raise ValueError(INVALID_AYAH)
            verdict = run.RIGHT if reference == self._under_test[key] else run.OTHER
        elif choice in (run.SILENCE, run.NOT_RECITATION, run.UNCLEAR):
            verdict = choice
        else:
            raise ValueError(f"{choice!r} is not a page choice")
        saved = {**{field: row.get(field) for field in (
            "split", "shard_file", "row", "audio_path", "audio_sha256")},
            "verdict": verdict, "heard_ayah": heard_ayah, "ayah": reference, "note": note}
        with self._lock:
            if stored_answer(self.verdicts.get(key)) != previous:
                raise StaleAnswer("this item was answered elsewhere since; reload the page")
            proposed = {**self.verdicts, key: saved}
            write_text_atomically(
                self.verdicts_path,
                "".join(json.dumps(proposed[item], ensure_ascii=False, sort_keys=True) + "\n"
                        for item in sorted(proposed)))
            self.verdicts = proposed
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
        if not isinstance(payload, dict):
            self.send_json({"error": "body must be {key, choice, ayah, note, previous}"},
                           status=400)
            return
        try:
            saved = self.state.record(str(payload.get("key", "")), str(payload.get("choice", "")),
                                      payload.get("ayah"), str(payload.get("note", "")),
                                      payload.get("previous"))
        except StaleAnswer as error:
            self.send_json({"error": str(error)}, status=409)
            return
        except KeyError:
            self.send_json({"error": "unknown item"}, status=404)
            return
        except ValueError as error:
            self.send_json({"error": str(error)}, status=400)
            return
        self.send_json({"progress": self.state.payload()["progress"],
                        "verdict": saved["verdict"], "ayah": saved["ayah"],
                        "heard_ayah": saved["heard_ayah"]})


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

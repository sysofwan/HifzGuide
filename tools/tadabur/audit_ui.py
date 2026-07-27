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
import sqlite3
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from training.whole_clip_audit import WholeClipAudit, build_whole_clip_audit

from . import eval_fixtures
from .audit_http import AuditHandler, serve
from .audit_sampler import WorklistItem
from .contrast_attribution import MARGINAL_CONTRAST, all_contrasts
from .eval_fixtures import ACCEPT, REJECT, EvalFixtureEntry
from .manifest import read_records
from .normalization import normalize_phonemes
from .signoff_results import build_signoff_view
from .smith_waterman import smith_waterman

_PAGE_PATH = Path(__file__).parent / "audit_ui_page.html"
_WHOLE_CLIP_PAGE_PATH = Path(__file__).parent / "whole_clip_audit_page.html"
_SIGNOFF_PAGE_PATH = Path(__file__).parent / "signoff_page.html"


@dataclass(frozen=True)
class SignoffReports:
    """Paths to the three fine-tune sign-off report artifacts (#37, helper for #10).

    Each is optional — the E ablation ladder, the F2 event eval, and the H integration eval are
    produced by separate offline runs, and the sign-off view renders whichever exist. ``enabled``
    is true once any is provided, which is what turns the ``/sign-off`` view on in the UI.
    """

    ladder: Path | None = None
    event_eval: Path | None = None
    integration: Path | None = None

    @property
    def enabled(self) -> bool:
        return any((self.ladder, self.event_eval, self.integration))

# The canonical quran.db (source of Uthmani ayah text), at the repo-root data/.
DEFAULT_QURAN_DB = Path(__file__).parents[2] / "data" / "quran.db"

# Bucket order shown in the UI: the seven attribution buckets, then marginal.
CONTRAST_ORDER: tuple[str, ...] = all_contrasts() + (MARGINAL_CONTRAST,)


def uthmani_index(quran_db_path: Path, surah_ayahs: set[str]) -> dict[str, str]:
    """Map ``"surah:ayah"`` -> Uthmani ayah text from ``quran.db``.

    Only the ``surah_ayah`` keys the worklist actually needs are looked up so the
    reviewer can read the true ayah while grading. Missing or malformed keys are
    skipped (the UI just shows no text for them) rather than failing the audit.
    """
    index: dict[str, str] = {}
    if not quran_db_path.is_file():
        return index
    with sqlite3.connect(quran_db_path) as conn:
        for key in surah_ayahs:
            try:
                surah, ayah = (int(part) for part in key.split(":"))
            except ValueError:
                continue
            row = conn.execute(
                "SELECT text FROM ayahs WHERE surah = ? AND ayah = ?", (surah, ayah)
            ).fetchone()
            if row is not None:
                index[key] = row[0]
    return index



def raw_reference_index(quran_db_path: Path, surah_ayahs: set[str]) -> dict[str, str]:
    """Map ``"surah:ayah"`` -> the *raw* reference phoneme string from ``quran.db``.

    Unlike :func:`reference_phoneme_index` (which returns the normalized string the
    gate scores against), this returns the full phonetization with madd length and
    idgham/ghunna markers intact — what a human needs to judge tajweed by ear. Read
    from the ``ayahs.phonemes`` column (the same DB used for Uthmani text). Missing
    or malformed keys are skipped rather than failing the audit.
    """
    index: dict[str, str] = {}
    if not quran_db_path.is_file():
        return index
    with sqlite3.connect(quran_db_path) as conn:
        for key in surah_ayahs:
            try:
                surah, ayah = (int(part) for part in key.split(":"))
            except ValueError:
                continue
            row = conn.execute(
                "SELECT phonemes FROM ayahs WHERE surah = ? AND ayah = ?", (surah, ayah)
            ).fetchone()
            if row is not None:
                index[key] = row[0]
    return index


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


def predicted_phoneme_index(manifest_path: Path) -> dict[str, str]:
    """Map ``clip_id`` -> the model's decoded phoneme string, from the manifest.

    Lets the audit UI show what the model actually *heard* for each clip so the
    reviewer can see where it diverged from the reference. Empty for clips whose
    manifest predates the ``predicted_phonemes`` field.
    """
    return {r.audio_filename: r.predicted_phonemes for r in read_records(manifest_path)}


def reference_phoneme_index(surah_ayahs: set[str]) -> dict[str, str]:
    """Map ``"surah:ayah"`` -> the normalized reference phoneme string.

    Loads the warm reference cache — the exact strings the ``.balanced`` gate
    scores against — so the UI's reference/predicted diff matches what admitted
    the clip. Degrades to ``{}`` if the cache is unavailable (rather than
    triggering a slow rebuild inside the server), leaving the UI to simply omit
    the reference line.
    """
    try:
        from .reference_phonemes import load_reference_phonemes

        references = load_reference_phonemes()
    except Exception:
        return {}
    return {key: references[key] for key in surah_ayahs if key in references}


def segment_display_index(
    segment_manifest_path: Path,
) -> dict[str, dict[str, dict[str, str]]]:
    """Per-clip (per-segment) display indexes from a scored segment manifest.

    In waqf-segment audit mode the audit unit is a segment, not a whole ayah, so
    its Uthmani text and realized (waqf-aware) reference differ from the ayah's and
    must be keyed by the per-segment ``clip_id`` rather than ``surah:ayah``. Reads
    :mod:`tadabur.segment_score`'s manifest — whose rows carry those per-segment
    display fields alongside the :class:`~tadabur.manifest.ManifestRecord` ones —
    and returns the ``surah_ayah`` / ``predicted`` / ``reference`` /
    ``raw_reference`` / ``uthmani`` maps :func:`main` feeds the server, each keyed
    by ``clip_id`` (the segment id).
    """
    out = {k: {} for k in ("surah_ayah", "predicted", "reference", "raw_reference", "uthmani")}
    with open(segment_manifest_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            clip_id = row["audio_filename"]
            out["surah_ayah"][clip_id] = row.get("surah_ayah", "")
            out["predicted"][clip_id] = row.get("predicted_phonemes", "")
            out["reference"][clip_id] = row.get("reference_phonemes", "")
            out["raw_reference"][clip_id] = row.get("raw_reference_phonemes", "")
            out["uthmani"][clip_id] = row.get("uthmani", "")
    return out


def align_phonemes(predicted: str, reference: str) -> list[dict[str, str]]:
    """Align ``predicted`` against ``reference`` into per-column diff cells.

    Runs the same normalization + Smith-Waterman the gate uses: ``predicted`` is
    the model's raw decode and is normalized here, while ``reference`` must
    already be normalized (the cache form) and is used verbatim — normalization
    is not idempotent, so re-normalizing it would collapse its shadda doubling
    and drop madd markers from the diff. Turns the recovered local alignment into
    a list of ``{"ref", "pred", "kind"}`` columns the browser renders as a
    two-row diff. ``kind`` is ``match`` (same phoneme), ``sub`` (different phoneme
    heard), ``del`` (reference phoneme dropped) or ``ins`` (extra phoneme
    inserted). Returns ``[]`` when either side is empty.
    """
    query = normalize_phonemes(predicted).normalized
    ref = reference
    if not query.strip() or not ref.strip():
        return []
    columns: list[dict[str, str]] = []
    for col in smith_waterman(query=query, reference=ref).columns:
        q, r = col.query_char, col.ref_char
        if q is None:
            kind = "del"
        elif r is None:
            kind = "ins"
        elif q == r:
            kind = "match"
        else:
            kind = "sub"
        columns.append({"ref": r or "", "pred": q or "", "kind": kind})
    return columns


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


def item_view(server: "AuditServer", item: WorklistItem) -> dict[str, object]:
    """The JSON shape one worklist row is sent to the browser as.

    Bundles the clip's identity, its Uthmani ayah text, and a reference-vs-
    predicted phoneme alignment so the reviewer can both read the true ayah and
    see exactly where the model's decode diverged before grading B vs C.
    """
    surah_ayah = server.surah_ayah.get(item.clip_id, "")
    predicted = server.predicted.get(item.clip_id, "")
    reference = server.reference.get(item.clip_id)
    if reference is None:
        reference = server.reference.get(surah_ayah, "")
    raw_reference = server.raw_reference.get(item.clip_id)
    if raw_reference is None:
        raw_reference = server.raw_reference.get(surah_ayah, "")
    uthmani = server.uthmani.get(item.clip_id)
    if uthmani is None:
        uthmani = server.uthmani.get(surah_ayah, "")
    return {
        "clip_id": item.clip_id,
        "contrast": item.contrast,
        "match_ratio": item.match_ratio,
        "surah_ayah": surah_ayah,
        "uthmani": uthmani,
        "reference_phonemes": reference,
        "raw_reference_phonemes": raw_reference,
        "predicted_phonemes": predicted,
        "alignment": align_phonemes(predicted, reference),
        "audio_url": f"/audio/{item.local_audio_path}",
        "audio_available": (server.audio_dir / item.local_audio_path).is_file(),
        "verdict": server.store.verdict_of(item.clip_id, item.contrast),
        "note": server.store.note_of(item.clip_id, item.contrast),
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
        uthmani: dict[str, str] | None = None,
        predicted: dict[str, str] | None = None,
        reference: dict[str, str] | None = None,
        raw_reference: dict[str, str] | None = None,
        whole_clip_audit: WholeClipAudit | None = None,
        signoff_reports: SignoffReports | None = None,
    ) -> None:
        self.items = items
        self.surah_ayah = surah_ayah
        self.store = store
        self.audio_dir = audio_dir
        self.uthmani = uthmani or {}
        self.predicted = predicted or {}
        self.reference = reference or {}
        self.raw_reference = raw_reference or {}
        self.whole_clip_audit = whole_clip_audit
        self.signoff_reports = signoff_reports or SignoffReports()
        self._by_key = {(i.clip_id, i.contrast): i for i in items}

    def state(self) -> dict[str, object]:
        """The full UI payload: every row's view plus per-contrast stats."""
        return {
            "items": [item_view(self, i) for i in self.items],
            "stats": contrast_stats(self.items, self.store),
            "contrast_order": list(CONTRAST_ORDER),
            "whole_clip_available": self.whole_clip_audit is not None,
            "signoff_available": self.signoff_reports.enabled,
        }

    def signoff_state(self) -> dict[str, object]:
        """The fine-tune sign-off results payload (#37) — the go/no-go for the #10 HITL gate.

        ``available`` is false when the UI was launched without any sign-off report, so the page
        can explain how to enable it. When available the three report artifacts are read *fresh*
        each request (via :func:`signoff_results.build_signoff_view`), so re-running an eval and
        refreshing shows the new numbers without restarting the server.
        """
        reports = self.signoff_reports
        if not reports.enabled:
            return {"available": False}
        view = build_signoff_view(reports.ladder, reports.event_eval, reports.integration)
        return {"available": True, **view}

    def whole_clip_state(self) -> dict[str, object]:
        """The whole-clip data-path payload for the read-only training-data view.

        ``available`` is false when the UI was launched without ``--clip-status`` (no
        whole-clip reconstruction), so the page can explain how to enable it rather than
        render an empty list. When available it carries every clip's view (each a plain
        nested dict via :func:`dataclasses.asdict`) and the training-eligibility summary.
        """
        audit = self.whole_clip_audit
        if audit is None:
            return {"available": False}
        return {
            "available": True,
            "clips": [asdict(view) for view in audit.views],
            "summary": {
                "clips_included": audit.clips_included,
                "clips_excluded": audit.clips_excluded,
                "exclusions_by_reason": audit.exclusions_by_reason,
            },
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


class _Handler(AuditHandler):
    """Routes ``/`` (page), ``/api/state``, ``/api/label`` and ``/audio/<file>``."""

    state: AuditServer  # bound onto the subclass by serve()

    def do_GET(self) -> None:
        path = unquote(urlparse(self.path).path)
        if path == "/":
            self.send_bytes(_PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/whole-clip":
            self.send_bytes(_WHOLE_CLIP_PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/sign-off":
            self.send_bytes(_SIGNOFF_PAGE_PATH.read_bytes(), "text/html; charset=utf-8")
        elif path == "/api/state":
            self.send_json(self.state.state())
        elif path == "/api/whole-clip":
            self.send_json(self.state.whole_clip_state())
        elif path == "/api/sign-off":
            self.send_json(self.state.signoff_state())
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
    parser.add_argument("--manifest", type=Path, default=None,
                        help="Filter manifest (full-ayah mode; for surah:ayah + predicted).")
    parser.add_argument("--segment-manifest", type=Path, default=None,
                        help="Scored segment manifest from tadabur.segment_score "
                             "(waqf-segment mode; per-segment reference/uthmani/predicted).")
    parser.add_argument("--clip-status", type=Path, default=None,
                        help="Per-clip status sidecar from tadabur.segment_score. With "
                             "--segment-manifest, enables the read-only /whole-clip view that "
                             "reconstructs the whole-clip training data path C (#25) feeds.")
    parser.add_argument("--audio-dir", type=Path, required=True, help="Directory of exported clip audio.")
    parser.add_argument("--accept", type=Path, default=eval_fixtures.SHOULD_ACCEPT_PATH,
                        help="should-accept fixture file to write (default: canonical path).")
    parser.add_argument("--reject", type=Path, default=eval_fixtures.SHOULD_REJECT_PATH,
                        help="should-reject fixture file to write (default: canonical path).")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (default: 127.0.0.1; use 0.0.0.0 to expose on the LAN).")
    parser.add_argument("--quran-db", type=Path, default=DEFAULT_QURAN_DB,
                        help="quran.db for Uthmani ayah text (default: repo data/quran.db).")
    parser.add_argument("--ladder-report", type=Path, default=None,
                        help="ablation-ladder report JSON (training.ablation_ladder report; E). "
                             "Enables the read-only /sign-off results view (#37, helper for #10).")
    parser.add_argument("--event-eval-report", type=Path, default=None,
                        help="waqf event-eval report JSON (tadabur.waqf_event_eval; F2) for /sign-off.")
    parser.add_argument("--integration-report", type=Path, default=None,
                        help="conditional-reference integration-eval report JSON (#35, H) for /sign-off.")
    args = parser.parse_args()

    items = load_worklist(args.worklist)
    store = LabelStore.load(args.accept, args.reject)
    if args.segment_manifest is not None:
        # Waqf-segment mode: display fields are per-segment (keyed by clip_id).
        idx = segment_display_index(args.segment_manifest)
        surah_ayah = idx["surah_ayah"]
        predicted = idx["predicted"]
        reference = idx["reference"]
        raw_reference = idx["raw_reference"]
        uthmani = idx["uthmani"]
    else:
        if args.manifest is None:
            parser.error("one of --manifest (full-ayah) or --segment-manifest is required")
        surah_ayah = surah_ayah_index(args.manifest)
        predicted = predicted_phoneme_index(args.manifest)
        reference = reference_phoneme_index(set(surah_ayah.values()))
        uthmani = uthmani_index(args.quran_db, set(surah_ayah.values()))
        raw_reference = raw_reference_index(args.quran_db, set(surah_ayah.values()))

    whole_clip_audit = None
    if args.clip_status is not None:
        if args.segment_manifest is None:
            parser.error("--clip-status requires --segment-manifest (whole-clip view).")
        whole_clip_audit = build_whole_clip_audit(args.segment_manifest, args.clip_status)

    server_state = AuditServer(
        items, surah_ayah, store, args.audio_dir, uthmani, predicted, reference,
        raw_reference, whole_clip_audit,
        SignoffReports(args.ladder_report, args.event_eval_report, args.integration_report),
    )

    httpd = serve(_Handler, server_state, args.port, args.host)
    labelled = sum(1 for i in items if store.verdict_of(i.clip_id, i.contrast))
    print(f"Loaded {len(items)} worklist rows ({labelled} already labelled); "
          f"{len(uthmani)} ayat with Uthmani text, {len(reference)} with reference phonemes.")
    if whole_clip_audit is not None:
        print(f"Whole-clip data path: {whole_clip_audit.clips_included} clips feed training, "
              f"{whole_clip_audit.clips_excluded} excluded "
              f"({whole_clip_audit.exclusions_by_reason}) — /whole-clip")
    if server_state.signoff_reports.enabled:
        print("Fine-tune sign-off results (E/F2/H) — /sign-off")
    print(f"Audit UI on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

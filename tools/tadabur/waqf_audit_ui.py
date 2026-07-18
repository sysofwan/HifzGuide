"""Local web UI for the P7.F0 waqf event-adjudication gate (#27).

Serves the candidate-boundary worklist (:mod:`tadabur.waqf_event_sampler`) in the
browser so a human can play each **whole** clip, seek to a candidate boundary, and
mark it — the analogue of the poison-audit UI (:mod:`tadabur.audit_ui`) for the
event-level signal ADR-0004 needs. A silence VAD cannot tell a real **waqf** from a
mid-word stop-consonant/hamza **closure**, nor confirm that a continued boundary was
a genuine **wasl**; the reviewer calls each, and the verdict is persisted straight
into the canonical waqf event-fixture JSONL (:mod:`tadabur.waqf_event_fixtures`), so
the UI resumes from — and is interchangeable with — whatever that file already holds.

Each **clip** is one review unit. Its Uthmani ayah text (which the manifest omits) is
recovered from ``quran.db`` for context, and the clip audio is served from ``--audio-dir``
— the whole-clip staging directory :mod:`tadabur.waqf_segments` writes, where each clip
already lives under its raw ``audio_filename`` (the candidate's ``audio_ref``). Assume-correct
by default: every boundary keeps its VAD-derived ``predicted`` class unless the reviewer
overrides it (a false positive → ``wasl``, a false negative → a stop). Only overrides are
persisted (:mod:`tadabur.waqf_event_fixtures`), plus a per-clip *reviewed* flag; ground truth
for the eval is each reviewed clip's predicted labels ⊕ those corrections.

Usage:
  python -m tadabur.waqf_audit_ui --candidates waqf_candidates.jsonl \\
    --clips waqf_clip_worklist.jsonl --audio-dir clips/ [--port 8000] [--host 0.0.0.0]

  ``--audio-dir`` is the same directory ``tadabur.waqf_segments`` staged the whole
  passing clips into (the audio the VAD/segmentation pass analysed to propose these
  candidate boundaries); no separate audio-export step is needed.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

from .audit_http import AuditHandler, serve
from .clip_status import read_clip_status
from .waqf_candidates import BOUNDARY_MATCH_TOL_S
from .waqf_event_fixtures import (
    MID_WORD_CLOSURE,
    WAQF,
    WAQF_EVENT_CLASSES,
    WAQF_EVENTS_PATH,
    WASL,
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

# ---- assumed-correct baseline: the full candidate manifest ---------------------

def load_candidates_by_clip(path: Path) -> dict[str, list[dict]]:
    """Group the full candidate manifest (JSONL of ``WaqfCandidate`` rows) by clip id.

    This is the *assumed-correct baseline*: every word edge of every clip, each carrying
    the VAD-derived ``predicted`` class (``waqf`` / ``mid_word_closure`` where a pause was
    found, ``wasl`` otherwise). In the correction-based model the reviewer overrides only
    the boundaries they judge wrong — a **false positive** (a predicted stop the reciter did
    not stop at → ``wasl``) or a **false negative** (a predicted ``wasl`` the reciter did stop
    at → ``waqf`` / ``mid_word_closure``); everything untouched keeps its predicted class.
    Rows are ordered by ``boundary_index``.
    """
    by_clip: dict[str, list[dict]] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            by_clip.setdefault(row["clip_id"], []).append(row)
    for rows in by_clip.values():
        rows.sort(key=lambda r: r["boundary_index"])
    return by_clip


def load_clip_worklist(path: Path) -> list[str]:
    """Read the sampled clip review-list (JSONL of ``{"clip_id": ...}``), in order, de-duped.

    The eval-set sample is now a set of *clips* to review end-to-end (not a set of sampled
    boundaries): the review unit that makes assume-correct-by-default trustworthy.
    """
    clips: list[str] = []
    seen: set[str] = set()
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            cid = json.loads(line)["clip_id"]
            if cid not in seen:
                seen.add(cid)
                clips.append(cid)
    return clips


def uthmani_words_index(surah_ayat: set[str]) -> dict[str, str]:
    """Display Uthmani text keyed to the SAME tokenization ``word_index`` counts against.

    Boundary ``word_index`` is defined over quran-transcript's ``Aya(...).uthmani_words``
    — the token stream the segmentation pass indexed. quran.db's Uthmani instead carries
    mushaf ornaments (e.g. the ۞ rub-el-hizb) as leading space-delimited tokens, so
    rendering it would shift every waqf/closure marker one word off its stop. We build the
    display string from those same ``uthmani_words``, space-joined, so a front-end split on
    whitespace reproduces the exact index basis and markers land on the right word.
    """
    from .waqf_segments import _uthmani_words

    return {sa: " ".join(_uthmani_words(sa)) for sa in surah_ayat}


@dataclass
class WaqfEventStore:
    """The reviewer's **overrides** to the assumed-correct baseline, persisted as fixtures.

    Only boundaries whose ground truth differs from the predicted class are stored — one
    :class:`WaqfEventEntry` per correction, keyed by ``(clip_id, boundary_index)``. Untouched
    boundaries keep their predicted class (assume-correct-by-default), so the fixture file
    holds just the corrections. :meth:`set` and :meth:`clear` rewrite the file atomically
    through the schema module, so the on-disk fixtures always equal the UI state and a restart
    resumes exactly where the human left off.
    """

    path: Path
    entries: dict[BoundaryKey, WaqfEventEntry]

    @classmethod
    def load(cls, path: Path) -> "WaqfEventStore":
        """Build a store from any already-recorded corrections in the fixture file."""
        entries = {(e.clip_id, e.boundary_index): e for e in load_waqf_events(path)}
        return cls(path, entries)

    def verdict_of(self, key: BoundaryKey) -> str | None:
        entry = self.entries.get(key)
        return entry.verdict if entry else None

    def note_of(self, key: BoundaryKey) -> str:
        entry = self.entries.get(key)
        return entry.note if entry else ""

    def live_entry(self, row: dict) -> WaqfEventEntry | None:
        """The override recorded for a live candidate ``row``, honoured only when it still
        describes the **same word edge at the same instant**.

        ``boundary_index`` orders the candidate boundaries within a clip and is re-assigned
        every time the candidates are re-segmented, so a stored override can end up sharing a
        slot with a *different* event that now occupies that index. Matching on it alone (or
        even ``+ word_index``) is unsafe: a re-read clip can carry several boundaries on the
        **same** Uthmani word — a stop and its re-read wasl — so identity also needs the
        boundary's timing. We require the stored ``word_index`` to match *and* the stored
        ``(start_s, end_s)`` to agree within :data:`BOUNDARY_MATCH_TOL_S` (candidate timing is
        deterministic, so a genuine match is essentially exact — the tolerance only absorbs the
        JSON float round-trip). A relocated boundary therefore reads as un-reviewed until a
        human re-confirms it, instead of silently attaching an old verdict to a new boundary.
        This is the single join both the per-clip view and the correction tally use, so they
        can never disagree about what is a live correction.
        """
        entry = self.entries.get((row["clip_id"], row["boundary_index"]))
        if entry is None or entry.word_index != row["word_index"]:
            return None
        if (abs(entry.start_s - row["start_s"]) > BOUNDARY_MATCH_TOL_S
                or abs(entry.end_s - row["end_s"]) > BOUNDARY_MATCH_TOL_S):
            return None
        return entry

    def set(self, entry: WaqfEventEntry) -> None:
        """Record (or overwrite) a boundary's correction and persist the fixture set.

        The change is staged in a copy and the on-disk file rewritten *before*
        ``self.entries`` is swapped in, so a rejected entry (e.g. an invalid verdict class
        the schema refuses to write) leaves both the store and the fixture file exactly as
        they were — the review session never holds a line that could not be persisted.
        """
        staged = dict(self.entries)
        staged[(entry.clip_id, entry.boundary_index)] = entry
        self._persist(staged)
        self.entries = staged

    def clear(self, key: BoundaryKey) -> None:
        """Drop a boundary's override (back to its assumed-correct predicted class) and persist."""
        if key not in self.entries:
            return
        staged = dict(self.entries)
        del staged[key]
        self._persist(staged)
        self.entries = staged

    def _persist(self, entries: dict[BoundaryKey, WaqfEventEntry]) -> None:
        ordered = sorted(entries.values(), key=lambda e: (e.clip_id, e.boundary_index))
        write_waqf_events(ordered, self.path)


def reviewed_path_for(fixtures: Path) -> Path:
    """Sibling file that records which clips are reviewed (beside the fixtures JSONL)."""
    return fixtures.with_name("waqf_reviewed_clips.json")


def flagged_path_for(fixtures: Path) -> Path:
    """Sibling file that records clips flagged for a later revisit (beside the fixtures JSONL)."""
    return fixtures.with_name("waqf_flagged_clips.json")


@dataclass
class ReviewedClipStore:
    """Which clips the reviewer has confirmed reviewed end-to-end.

    A clip becomes an eval unit only once it is reviewed: its untouched boundaries can then
    be trusted as *confirmed* predicted labels (assume-correct-by-default), not merely unseen.
    Persisted as a small JSON list beside the fixtures so a restart resumes the review scope.
    """

    path: Path
    clips: set[str]

    @classmethod
    def load(cls, path: Path) -> "ReviewedClipStore":
        clips: set[str] = set()
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
            clips = set(data.get("reviewed", []))
        return cls(path, clips)

    def is_reviewed(self, clip_id: str) -> bool:
        return clip_id in self.clips

    def set_reviewed(self, clip_id: str, reviewed: bool) -> None:
        if reviewed:
            self.clips.add(clip_id)
        else:
            self.clips.discard(clip_id)
        self._persist()

    def _persist(self) -> None:
        payload = {"reviewed": sorted(self.clips)}
        self.path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


@dataclass
class FlaggedClipStore:
    """Clips the reviewer flagged to revisit later, each with a free-text comment.

    Independent of the reviewed/override state: flagging is a personal "come back to this"
    bookmark (an unclear stop, ambiguous audio, a suspected data issue) and does not affect
    the eval set. Persisted as ``{clip_id: comment}`` beside the fixtures so flags survive a
    restart. An empty comment is allowed; flagging with a blank comment still bookmarks the clip.
    """

    path: Path
    comments: dict[str, str]

    @classmethod
    def load(cls, path: Path) -> "FlaggedClipStore":
        comments: dict[str, str] = {}
        if path.is_file():
            data = json.loads(path.read_text(encoding="utf-8") or "{}")
            comments = dict(data.get("flagged", {}))
        return cls(path, comments)

    def is_flagged(self, clip_id: str) -> bool:
        return clip_id in self.comments

    def comment_of(self, clip_id: str) -> str:
        return self.comments.get(clip_id, "")

    def set_flagged(self, clip_id: str, flagged: bool, comment: str = "") -> None:
        if flagged:
            self.comments[clip_id] = comment
        else:
            self.comments.pop(clip_id, None)
        self._persist()

    def _persist(self) -> None:
        payload = {"flagged": {c: self.comments[c] for c in sorted(self.comments)}}
        self.path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")


def review_stats(server: "WaqfAuditServer") -> dict[str, int]:
    """Clip-review progress plus the correction tallies ADR-0004's eval reads.

    Corrections are classified against the **current** candidate baseline: a predicted stop
    the reviewer calls ``wasl`` is a **false positive**; a predicted ``wasl`` called a stop is
    a **false negative**; a ``waqf`` ↔ ``mid_word_closure`` swap is a class fix. The predicted
    class is read live from the candidate rows, not the (possibly stale) snapshot stored with
    the override, and each override is resolved through :meth:`WaqfEventStore.live_entry` so a
    relocated boundary is ignored — re-segmenting the candidates keeps the tally honest and
    identical to the per-clip correction list the page navigates by.
    """
    stops = {WAQF, MID_WORD_CLOSURE}
    fp = fn = class_fix = 0
    for row in server._boundary_rows.values():
        entry = server.store.live_entry(row)
        if entry is None:
            continue
        pred, truth = row["predicted"], entry.verdict
        if pred == truth:
            continue
        if pred in stops and truth == WASL:
            fp += 1
        elif pred == WASL and truth in stops:
            fn += 1
        else:
            class_fix += 1
    return {
        "clips_total": len(server.clips),
        "clips_reviewed": sum(1 for c in server.clips if server.reviewed.is_reviewed(c)),
        "clips_flagged": sum(1 for c in server.clips if server.flagged.is_flagged(c)),
        "false_positive": fp,
        "false_negative": fn,
        "class_fix": class_fix,
    }


def boundary_view(row: dict, verdict: str | None, note: str) -> dict[str, object]:
    """One word-edge boundary as the UI sees it: its span, predicted class, and any override.

    ``verdict`` is the reviewer's override (``None`` when untouched — the boundary keeps its
    predicted class as assumed-correct ground truth). ``truth`` is the effective ground-truth
    class the UI marks the ayah / timeline with: the override if present, else the prediction.
    """
    return {
        "boundary_index": row["boundary_index"],
        "word_index": row["word_index"],
        "start_s": row["start_s"],
        "end_s": row["end_s"],
        "predicted": row["predicted"],
        "verdict": verdict,
        "truth": verdict if verdict is not None else row["predicted"],
        "note": note,
    }


def clip_view(server: "WaqfAuditServer", clip_id: str) -> dict[str, object]:
    """One clip page: the whole recitation, every word-edge boundary, and its reviewed flag.

    ``boundaries`` is the clip's full candidate set (every word edge) with each boundary's
    predicted class, current override, and effective ``truth`` — the reviewer plays the clip
    once and only flips the exceptions. Overrides are resolved through
    :meth:`WaqfEventStore.live_entry`, so a verdict left on a boundary that re-segmentation
    has since relocated is dropped (the boundary shows as un-reviewed) rather than mislabelled.
    Audio is served under the clip's staged filename. ``recited_words`` (from the clip-status
    sidecar, ``None`` when absent) is how many leading Uthmani words the reciter actually
    recited; the page hides the never-recited tail of an early-stop clip so it draws no
    phantom markers.
    """
    rows = server.candidates_by_clip.get(clip_id, [])
    first = rows[0] if rows else None
    audio_ref = first["audio_ref"] if first else clip_id
    surah_ayah = first["surah_ayah"] if first else ""
    boundaries = []
    for r in rows:
        entry = server.store.live_entry(r)
        boundaries.append(boundary_view(
            r,
            entry.verdict if entry else None,
            entry.note if entry else "",
        ))
    return {
        "clip_id": clip_id,
        "surah_ayah": surah_ayah,
        "uthmani": server.uthmani.get(surah_ayah, ""),
        "audio_url": f"/audio/{audio_ref}",
        "audio_available": (server.audio_dir / audio_ref).is_file(),
        "reviewed": server.reviewed.is_reviewed(clip_id),
        "flagged": server.flagged.is_flagged(clip_id),
        "flag_comment": server.flagged.comment_of(clip_id),
        "recited_words": server.recited_words_by_clip.get(clip_id),
        "boundaries": boundaries,
    }


class WaqfAuditServer:
    """Holds the review clip list, the candidate baseline, Uthmani index and both stores.

    A thin state object the request handler reads; keeps the handler free of globals and
    makes the request logic unit-testable in isolation.
    """

    def __init__(
        self,
        clips: list[str],
        candidates_by_clip: dict[str, list[dict]],
        uthmani: dict[str, str],
        store: WaqfEventStore,
        reviewed: ReviewedClipStore,
        audio_dir: Path,
        flagged: FlaggedClipStore | None = None,
        recited_words_by_clip: dict[str, int | None] | None = None,
    ) -> None:
        self.clips = clips
        self.candidates_by_clip = candidates_by_clip
        self.uthmani = uthmani
        self.store = store
        self.reviewed = reviewed
        self.flagged = flagged if flagged is not None else FlaggedClipStore(Path(), {})
        self.audio_dir = audio_dir
        self.recited_words_by_clip = recited_words_by_clip or {}
        self._boundary_rows: dict[BoundaryKey, dict] = {
            (cid, r["boundary_index"]): r
            for cid, rows in candidates_by_clip.items() for r in rows
        }

    def state(self) -> dict[str, object]:
        """The full UI payload: one page per reviewed-or-pending clip, plus review stats."""
        return {
            "clips": [clip_view(self, c) for c in self.clips],
            "stats": review_stats(self),
            "classes": list(WAQF_EVENT_CLASSES),
        }

    def apply_label(self, payload: dict) -> dict[str, object]:
        """Set or clear one boundary's override against the assumed-correct baseline.

        A ``verdict`` of ``None`` — or one equal to the boundary's predicted class — clears
        the override, returning the boundary to its assumed-correct predicted class (so the
        fixtures never carry a redundant "confirmation" line). An unknown boundary — not in
        any reviewed/sampled clip's candidate set — is rejected. The stored entry recovers
        every non-verdict field from the candidate manifest, so the request is trusted only
        for the verdict and note.
        """
        key = (payload["clip_id"], payload["boundary_index"])
        row = self._boundary_rows.get(key)
        if row is None:
            raise KeyError(f"no candidate boundary for {key!r}")

        verdict = payload.get("verdict")
        if verdict is None or verdict == row["predicted"]:
            self.store.clear(key)
        else:
            self.store.set(WaqfEventEntry(
                clip_id=row["clip_id"],
                audio_ref=row["audio_ref"],
                surah_ayah=row["surah_ayah"],
                boundary_index=row["boundary_index"],
                word_index=row["word_index"],
                start_s=row["start_s"],
                end_s=row["end_s"],
                predicted=row["predicted"],
                verdict=verdict,
                note=payload.get("note", ""),
            ))
        return {"stats": review_stats(self)}

    def apply_review(self, payload: dict) -> dict[str, object]:
        """Mark (or unmark) a clip reviewed end-to-end — the flag that admits it to the eval set."""
        clip_id = payload["clip_id"]
        if clip_id not in self.candidates_by_clip:
            raise KeyError(f"unknown clip {clip_id!r}")
        self.reviewed.set_reviewed(clip_id, bool(payload.get("reviewed", True)))
        return {"stats": review_stats(self)}

    def apply_flag(self, payload: dict) -> dict[str, object]:
        """Flag (or clear) a clip for a later revisit, with an optional free-text comment.

        Independent of review/override state — a personal bookmark that never affects the eval
        set. ``flagged=False`` clears the flag and its comment; the comment defaults to blank.
        """
        clip_id = payload["clip_id"]
        if clip_id not in self.candidates_by_clip:
            raise KeyError(f"unknown clip {clip_id!r}")
        flagged = bool(payload.get("flagged", True))
        self.flagged.set_flagged(clip_id, flagged, str(payload.get("comment", "")))
        return {"stats": review_stats(self), "clip": clip_view(self, clip_id)}



class _Handler(AuditHandler):
    """Routes ``/`` (page), ``/api/state``, ``/api/label``, ``/api/review``, ``/api/flag`` and ``/audio/<file>``."""

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
        handlers = {
            "/api/label": self.state.apply_label,
            "/api/review": self.state.apply_review,
            "/api/flag": self.state.apply_flag,
        }
        handler = handlers.get(path)
        if handler is None:
            self.send_json({"error": "not found"}, status=404)
            return
        length = int(self.headers.get("Content-Length", 0))
        payload = json.loads(self.rfile.read(length) or b"{}")
        try:
            self.send_json(handler(payload))
        except (KeyError, ValueError) as exc:
            self.send_json({"error": str(exc)}, status=400)


def _review_clips(args, candidates_by_clip: dict[str, list[dict]]) -> list[str]:
    """The ordered set of clips to review: an explicit ``--clips`` list, else the distinct
    clips a legacy boundary ``--worklist`` sampled. Clips absent from the manifest are dropped."""
    if args.clips:
        return [c for c in load_clip_worklist(args.clips) if c in candidates_by_clip]
    ordered: list[str] = []
    seen: set[str] = set()
    for item in load_worklist(args.worklist):
        if item.clip_id in candidates_by_clip and item.clip_id not in seen:
            seen.add(item.clip_id)
            ordered.append(item.clip_id)
    return ordered


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--candidates", type=Path, required=True,
                        help="Full candidate manifest (tadabur.waqf_candidates) — the assumed-correct baseline.")
    parser.add_argument("--clips", type=Path, default=None,
                        help="Sampled clip review-list (JSONL of {\"clip_id\": ...}). The eval sample.")
    parser.add_argument("--worklist", type=Path, default=None,
                        help="Legacy boundary worklist; its distinct clips are reviewed when --clips is omitted.")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="Whole-clip staging dir from tadabur.waqf_segments (clips served by audio_filename).")
    parser.add_argument("--fixtures", type=Path, default=WAQF_EVENTS_PATH,
                        help="Waqf event-fixture file for corrections (default: canonical path).")
    parser.add_argument("--clip-status", type=Path, default=None,
                        help="Per-clip segmentation status sidecar (tadabur.clip_status) — supplies "
                             "recited_words so the page hides the never-recited tail of an early-stop clip.")
    parser.add_argument("--port", type=int, default=8000, help="Port to serve on (default: 8000).")
    parser.add_argument("--host", default="127.0.0.1",
                        help="Interface to bind (default: 127.0.0.1; use 0.0.0.0 to expose on the LAN).")
    args = parser.parse_args()
    if not args.clips and not args.worklist:
        parser.error("one of --clips or --worklist is required to select the review clips")

    candidates_by_clip = load_candidates_by_clip(args.candidates)
    clips = _review_clips(args, candidates_by_clip)
    selected = {c: candidates_by_clip[c] for c in clips}
    store = WaqfEventStore.load(args.fixtures)
    reviewed = ReviewedClipStore.load(reviewed_path_for(args.fixtures))
    flagged = FlaggedClipStore.load(flagged_path_for(args.fixtures))
    surah_ayat = {rows[0]["surah_ayah"] for rows in selected.values() if rows}
    uthmani = uthmani_words_index(surah_ayat)
    recited_words_by_clip = (
        {s.audio_filename: s.recited_words for s in read_clip_status(args.clip_status)}
        if args.clip_status else {}
    )
    server_state = WaqfAuditServer(
        clips, selected, uthmani, store, reviewed, args.audio_dir, flagged,
        recited_words_by_clip,
    )

    httpd = serve(_Handler, server_state, args.port, args.host)
    n_bounds = sum(len(rows) for rows in selected.values())
    n_reviewed = sum(1 for c in clips if reviewed.is_reviewed(c))
    n_flagged = sum(1 for c in clips if flagged.is_flagged(c))
    print(f"Loaded {len(clips)} review clips ({n_bounds} candidate boundaries; "
          f"{n_reviewed} already reviewed, {n_flagged} flagged); {len(uthmani)} ayat with Uthmani text.")
    print(f"Waqf audit UI on http://{args.host}:{args.port}  (Ctrl-C to stop)")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()

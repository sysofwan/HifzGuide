"""tlog S6's audit (#191): the strata draw, the listening session's staging and the report.

The pool pipeline is :mod:`tadabur.tlog_pool` (S6 is its first caller) and S6's population and
5% rule are :mod:`tadabur.tlog_ident_run`; this module is the audit half only, beside
:mod:`tadabur.tlog_audit_ui`, so S7 imports the pool without dragging the audit in.

It follows the frozen ``docs/tlog-holdout-preregistration.md`` ("Audit (S6)") as amended by
``docs/tlog-s6-amendment.md`` §4-§5 and the overlap clarification
``docs/tlog-s6-amendment-note-overlap.md``. The worklist is one JSONL row per drawn item:

    tools/tadabur/tlog/audit/worklist.jsonl
    tools/tadabur/tlog/audit/verdicts.jsonl

Usage, from ``tools/``::

    python -m tadabur.tlog_audit audit
    python -m tadabur.tlog_audit stage --audio-dir /root/scratch/tlog/s6/audio \\
        --cache-dir /root/scratch/tlog/s6/cache
    python -m tadabur.tlog_audit report
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from training.counterfactual_eval import wilson_interval
from training.tashkeel_eval import write_text_atomically

from . import tlog_ident
from .nonpro_sources import audio_sha256
from .tlog_ident import BOTH, FILENAME, IDENTIFIED, NONE, load_identifier
from .tlog_ident_run import AMENDMENT_APPROVAL, AMENDMENT_PATH, DURATION_BANDS, SALT, band_of
from .tlog_index import REPO_ROOT, read_jsonl, sha256_file, write_jsonl
from .tlog_overlap import iter_pool_rows
from .tlog_pass1 import read_jsonl_gz, shard_rows
from .tlog_pool import (IDENT_DIR, NOT_DECODED, WHOLE_CLIP, decode_limit, default_audio,
                        is_non_recitation, rank, read_ident_tree, row_key)

#: The pre-registered audit strata and sizes (frozen table; architect r20 Q3).
AUDIT_SIZES = {"both": 40, "relabelled": 30, "non_recitation": 25, "silence": 15,
               "unclean identified": 25, "below-bar disagreement": 15}
#: At least this many silence items come from the unclean split (architect r13).
SILENCE_UNCLEAN = 8
#: The frozen verdict vocabulary (pre-registration "Audit (S6)"), and r20 Q4's table.
RIGHT = "right ayah"
OTHER = "other ayah"
SILENCE = "silence"
NOT_RECITATION = "not recitation"
UNCLEAR = "unclear"
VERDICTS = (RIGHT, OTHER, SILENCE, NOT_RECITATION, UNCLEAR)
#: The overlap disclosure (r20 Q6), read from ``overlap/meta.json`` at render time.
OUTSIDE_ASR_LIMIT = 0.02
#: Below this many decided items an interval is not read (amendment §4).
MIN_READABLE_N = 10
#: The S5b §9 rule, pre-stated before any verdict: at this many audited
#: ``non_recitation`` clips heard as recitation, the 5% consequence pauses.
FLOOR_PAUSE_AT = 3
#: The commit the corrected full-frame worklist went live at, with zero answers. After it the
#: architect r22 fallback (a restricted worklist plus a supplemented sample) cannot arise.
FLOOR_REDRAW_COMMIT = "20e371c"
#: The PR whose ``refs/pull/<N>/head`` keeps every cited S6 SHA after the squash merge (I4.2).
PULL_REQUEST = 207

AUDIT_DIR = Path(__file__).parent / "tlog" / "audit"
DOC_PATH = REPO_ROOT / "docs" / "tlog-audit.md"
OVERLAP_META_PATH = Path(__file__).parent / "tlog" / "overlap" / "meta.json"
#: The dated clarification of the strata overlap (architect r22), cited by the report.
OVERLAP_NOTE_PATH = REPO_ROOT / "docs" / "tlog-s6-amendment-note-overlap.md"


@dataclass(frozen=True)
class AuditDraw:
    """The drawn audit items and each stratum's target/eligible/drawn/shortfall census."""

    items: tuple[dict, ...]
    strata: dict[str, dict[str, int]]


# --- the audit draw ----------------------------------------------------------------------------


#: Amendment §4 defines each stratum by its own predicate and sets no precedence between them: a
#: row satisfying two predicates is drawn in both and served once
#: (``docs/tlog-s6-amendment-note-overlap.md``).
def _matching_strata(row: Mapping[str, Any]) -> tuple[str, ...]:
    """Every stratum predicate this row satisfies, in :data:`AUDIT_SIZES` order.

    ``below-bar disagreement`` needs ``label_source == "filename"``: S5 records ``both``
    exactly when the best class and the filename class are one text class, so a ``both`` row is
    never a disagreement even when its two label strings differ (repeated ayat, e.g. ``26:190``
    and ``26:8`` are class 2925), and ``identified`` belongs to the relabel strata.
    """
    if row.get("decode_mode") == NOT_DECODED:
        return ()
    floor_ok = row.get("id_floor_ok")
    matched = set()
    if tuple(row.get("exclusions") or ()) == ("silence",):
        matched.add(SILENCE)
    if row.get("label_source") == BOTH:
        matched.add("both")
    if row["split"] == "clean" and row.get("filename_label") and \
            row.get("label_source") == IDENTIFIED:
        matched.add("relabelled")
    if row["split"] == "unclean" and row.get("label_source") == IDENTIFIED:
        matched.add("unclean identified")
    if row.get("decode_mode") == WHOLE_CLIP and is_non_recitation(row):
        matched.add("non_recitation")
    if (floor_ok is True and row.get("label_source") == FILENAME
            and row.get("class_label") != row.get("filename_label")):
        matched.add("below-bar disagreement")
    return tuple(stratum for stratum in AUDIT_SIZES if stratum in matched)


def _stratum_label(stratum: str, row: Mapping[str, Any]) -> str | None:
    """The ayah the stratum scores against (amendment §5)."""
    if stratum in ("relabelled", "unclean identified"):
        return row.get("assigned_label")
    if stratum in ("both", "below-bar disagreement"):
        return row.get("filename_label")
    return None


def _page_stratum(strata: Sequence[str], row: Mapping[str, Any]) -> str:
    """The one stratum whose label the served page treats as right for a shared row.

    The report never reads this field: it scores a row in every stratum it was drawn in. The
    page needs a single label under test for its right/other choice, and a labelled stratum's is
    the one the owner hears the filename with.
    """
    labelled = [stratum for stratum in strata if _stratum_label(stratum, row) is not None]
    return (labelled or list(strata))[0]


def _blocking_reason(row: Mapping[str, Any]) -> str:
    return "overlap" if row.get("overlaps_filename") else "delta"


def _stratum_members(records: Iterable[Mapping[str, Any]],
                     salt: str) -> dict[str, list[dict]]:
    """Each stratum's full eligible set, sorted by that stratum's salted rank.

    A row satisfying two predicates is in both lists: the strata overlap, and each draws from
    its own set.
    """
    members: dict[str, list[dict]] = defaultdict(list)
    for row in records:
        strata = _matching_strata(row)
        if not strata:
            continue
        item = {**dict(row),
                "blocking_reason": (_blocking_reason(row)
                                    if "below-bar disagreement" in strata else None)}
        for stratum in strata:
            members[stratum].append(item)
    return {stratum: sorted(rows, key=lambda r: rank(salt, stratum, *row_key(r)))
            for stratum, rows in members.items()}


def _served_item(item: Mapping[str, Any], drawn_in: Iterable[str]) -> dict:
    """One worklist row carrying the strata it was drawn in, in the frozen table's order.

    ``floor_sample`` marks the §9 sample: the ``non_recitation`` draw is exactly the frozen
    full-frame prefix, so every row the draw puts in that stratum is a sample row.
    """
    strata = [stratum for stratum in AUDIT_SIZES if stratum in set(drawn_in)]
    return {**dict(item), "strata": strata, "stratum": _page_stratum(strata, item),
            "floor_sample": "non_recitation" in strata}


def audit_draw(records: Iterable[Mapping[str, Any]], *, sizes: Mapping[str, int] | None = None,
               salt: str | None = None) -> AuditDraw:
    """Draw each stratum's salted prefix; a shortfall is recorded and its slots go to ``both``.

    Each stratum draws from its own full eligible set, so one row can be drawn in two strata and
    is served once.
    """
    salt = SALT if salt is None else salt
    sizes = dict(AUDIT_SIZES if sizes is None else sizes)
    if set(sizes) - set(AUDIT_SIZES):
        raise ValueError(f"sizes name unknown strata: {sorted(set(sizes) - set(AUDIT_SIZES))}")
    members = _stratum_members(records, salt)
    drawn: dict[str, list[dict]] = {}
    census: dict[str, dict[str, int]] = {}
    shortfall = 0
    for stratum in AUDIT_SIZES:
        if stratum == "both":
            continue
        eligible = members.get(stratum, [])
        target = sizes.get(stratum, 0)
        chosen = (_draw_silence(eligible, target, salt) if stratum == SILENCE
                  else eligible[:target])
        drawn[stratum] = chosen
        missing = max(0, target - len(chosen))
        shortfall += missing
        census[stratum] = {"target": target, "eligible": len(eligible),
                           "drawn": len(chosen), "shortfall": missing}
    both_target = sizes.get("both", 0) + shortfall
    eligible = members.get("both", [])
    drawn["both"] = eligible[:both_target]
    census["both"] = {"target": sizes.get("both", 0), "eligible": len(eligible),
                      "drawn": len(drawn["both"]),
                      "shortfall": max(0, both_target - len(drawn["both"]))}
    served: dict[str, dict] = {}
    drawn_in: dict[str, list[str]] = defaultdict(list)
    for stratum in AUDIT_SIZES:
        for item in drawn[stratum]:
            drawn_in[_item_key(item)].append(stratum)
    for stratum in AUDIT_SIZES:
        for item in drawn[stratum]:
            key = _item_key(item)
            if key not in served:
                served[key] = _served_item(item, drawn_in[key])
    return AuditDraw(items=tuple(served.values()), strata=census)


def _draw_silence(members: list[dict], target: int, salt: str) -> list[dict]:
    """At least ``SILENCE_UNCLEAN`` unclean, then clean, then unclean again if clean is short."""
    if target == 0:
        return []
    unclean = sorted((r for r in members if r["split"] == "unclean"),
                     key=lambda r: rank(salt, "silence", *row_key(r)))
    clean = sorted((r for r in members if r["split"] != "unclean"),
                   key=lambda r: rank(salt, "silence", *row_key(r)))
    first = unclean[:SILENCE_UNCLEAN]
    remaining = target - len(first)
    rest_clean = clean[:remaining]
    remaining -= len(rest_clean)
    extra = unclean[len(first):len(first) + remaining]
    return first + rest_clean + extra


def label_under_test(row: Mapping[str, Any]) -> str | None:
    """The ayah ``row["stratum"]`` scores against."""
    return _stratum_label(row["stratum"], row)


def verdict_outcome(item: Mapping[str, Any], verdict: Mapping[str, Any], *,
                    canonical: Callable[[str], str | None] | None = None) -> dict:
    """Score one verdict against the stratum's label under test (r20 Q4).

    ``unclear`` is an undecided outcome for every stratum: not hearing a clip is not a measured
    wrong ayah. When ``canonical`` is given (the report has the reference database), the heard
    ayah is compared by text class, so an identical recitation of a repeated ayah counts as the
    same ayah; without it, labels are compared literally.
    """
    heard = verdict.get("verdict")
    if heard not in VERDICTS:
        raise ValueError(f"{heard!r} is not one of {VERDICTS}")
    recitation = heard in (RIGHT, OTHER)
    label = label_under_test(item)
    heard_ayah = verdict.get("ayah")

    def same(first: object, second: object) -> bool | None:
        if first is None or second is None:
            return None
        if canonical is None:
            return first == second
        one, two = canonical(str(first)), canonical(str(second))
        return None if one is None or two is None else one == two

    if heard == UNCLEAR:
        return {"label_under_test": label, "correct": None, "best_correct": None,
                "recitation": False}
    if label is not None:
        matched = same(heard_ayah, label)
        correct = (heard == RIGHT) if matched is None else (recitation and matched)
        best_correct = None
        if item["stratum"] == "below-bar disagreement":
            best = same(heard_ayah, item.get("class_label"))
            best_correct = False if best is None else (recitation and best)
        return {"label_under_test": label, "correct": correct,
                "best_correct": best_correct, "recitation": recitation}
    if item["stratum"] == "non_recitation":
        return {"label_under_test": None, "correct": not recitation, "best_correct": None,
                "recitation": recitation}
    if item["stratum"] == SILENCE:
        return {"label_under_test": None, "correct": heard == SILENCE,
                "best_correct": None, "recitation": recitation}
    raise ValueError(f"no verdict rule for stratum {item['stratum']!r}")


def _item_key(row: Mapping[str, Any]) -> str:
    return hashlib.sha256(f"{row['split']}:{row['shard_file']}:{row['row']}".encode()
                          ).hexdigest()[:16]


def audit_payload(draw: AuditDraw, *, texts: Mapping[str, str] | None = None,
                  verdicts: Mapping[str, Mapping[str, Any]] | None = None,
                  canonical: Callable[[str], str | None] | None = None) -> dict:
    """The served payload: anonymous candidate texts, no stratum or attribution.

    With ``canonical``, labels that are the same text class are one candidate (the class's
    canonical label), so an identical recitation is not offered twice and either copy scores the
    same.
    """
    verdicts = verdicts or {}
    texts = texts or {}
    items = []
    for row in draw.items:
        key = _item_key(row)
        labels: list[str] = []
        for label in (row.get("filename_label"), row.get("class_label")):
            if not label:
                continue
            name = label if canonical is None else canonical(label)
            if name is not None and name not in labels:
                labels.append(name)
        labels.sort(key=lambda label: rank(SALT, "candidate", key, label))
        candidates = [{"ayah": label, "text": texts.get(label, "")} for label in labels]
        saved = verdicts.get(key)
        items.append({"key": key, "candidates": candidates,
                      "choices": [RIGHT, OTHER, SILENCE, NOT_RECITATION, UNCLEAR],
                      "heard": saved.get("verdict") if saved else None,
                      "heard_ayah": (saved.get("heard_ayah", saved.get("ayah"))
                                     if saved else None),
                      "note": saved.get("note", "") if saved else ""})
    answered = sum(1 for item in items if item["heard"])
    return {"items": items, "progress": {"answered": answered, "total": len(items)}}


# --- staging, verdicts and the generated page --------------------------------------------------


def _silence_candidates() -> list[dict]:
    rows = []
    for row in iter_pool_rows():
        if row["role"] == "look" and tuple(row["exclusions"]) == ("silence",):
            rows.append({**row, "label_source": NONE, "id_floor_ok": None,
                         "class_label": None, "assigned_label": None,
                         "overlaps_filename": None, "decode_mode": None})
    return rows


def audit_candidates(*, ident_dir: Path = IDENT_DIR) -> list[dict]:
    """The drawn population: ident look rows, ident sample rows, and silence-only look rows."""
    rows: list[dict] = []
    for split in ("clean", "unclean"):
        rows += [{**row, "population": "look", "role": "look"}
                 for row in read_jsonl_gz(ident_dir / "look" / f"{split}.jsonl.gz")]
    rows += [{**row, "population": "train_sample", "role": "train"}
             for row in read_jsonl_gz(ident_dir / "train_sample.jsonl.gz")]
    rows += [{**row, "population": "look"} for row in _silence_candidates()]
    return rows


def _wav_bytes(data: bytes) -> bytes:
    """The clip as 16 kHz mono PCM WAV, the format the listening page serves."""
    import io

    import soundfile as sf

    buffer = io.BytesIO()
    sf.write(buffer, default_audio(data), 16_000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def stage(draw: AuditDraw, *, audio_dir: Path, cache_dir: Path) -> None:
    """Stream the drawn rows' shards, verify each audio SHA-256 and keep the drawn clips.

    A clip the previous worklist staged and this one does not is deleted, so a redraw leaves no
    orphan audio behind.
    """
    wanted: dict[tuple[str, str], dict[int, dict]] = defaultdict(dict)
    for position, row in enumerate(draw.items):
        wanted[(row["split"], row["shard_file"])][row["row"]] = {**row, "index": position}
    audio_dir = Path(audio_dir)
    audio_dir.mkdir(parents=True, exist_ok=True)
    worklist: list[dict] = []
    for (split, stem), rows in sorted(wanted.items()):
        for index, item in enumerate(shard_rows(f"data/{stem}.parquet", Path(cache_dir))):
            row = rows.get(index)
            if row is None:
                continue
            data = item["audio"]["bytes"]
            checksum = audio_sha256(data)
            if checksum != row["audio_sha256"]:
                raise ValueError(f"{row_key(row)} is {checksum}, not {row['audio_sha256']}")
            name = f"{_item_key(row)}_{checksum[:8]}.wav"
            (audio_dir / name).write_bytes(_wav_bytes(data))
            worklist.append({**{key: row[key] for key in (
                "population", "split", "shard_file", "row", "role", "audio_path",
                "audio_sha256", "duration_s", "filename_label", "assigned_label",
                "class_label", "label_source", "id_floor_ok", "overlaps_filename",
                "blocking_reason", "strata", "stratum", "floor_sample")},
                "local_audio": name})
    if len(worklist) != len(draw.items):
        raise ValueError(f"staged {len(worklist)} of {len(draw.items)} drawn items")
    previous_path = Path(AUDIT_DIR) / "worklist.jsonl"
    if previous_path.is_file():
        kept = {row["local_audio"] for row in worklist}
        for previous in read_jsonl(previous_path):
            if previous["local_audio"] not in kept:
                (audio_dir / previous["local_audio"]).unlink(missing_ok=True)
    write_jsonl(worklist, Path(AUDIT_DIR) / "worklist.jsonl", sort_keys=True)


def read_verdicts(path: Path = AUDIT_DIR / "verdicts.jsonl") -> dict[str, dict]:
    path = Path(path)
    if not path.is_file():
        return {}
    verdicts: dict[str, dict] = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        row = json.loads(line)
        key = _item_key(row)
        if key in verdicts:
            raise ValueError(f"{path}:{lineno}: a second verdict for {key}")
        verdicts[key] = row
    return verdicts


def _precision(rows: Sequence[Mapping[str, Any]], verdicts: Mapping[str, Mapping[str, Any]],
               canonical: Callable[[str], str | None] | None = None) -> dict:
    scored = []
    for row in rows:
        verdict = verdicts.get(_item_key(row))
        if verdict is None:
            continue
        outcome = verdict_outcome(row, verdict, canonical=canonical)
        scored.append(outcome)
    decided = [outcome for outcome in scored if outcome["correct"] is not None]
    correct = sum(1 for outcome in decided if outcome["correct"])
    interval = list(wilson_interval(correct, len(decided))) if decided else None
    return {"n": len(scored), "decided": len(decided), "correct": correct,
            "precision": (correct / len(decided)) if decided else None,
            "wilson": interval,
            "recitation": sum(1 for outcome in scored if outcome["recitation"]),
            "best_correct": sum(1 for outcome in scored if outcome["best_correct"])}


def _interval_text(correct: int, n: int) -> str:
    """The amendment §4 rule: an interval at any n, unreadable below ten decided items."""
    if n == 0:
        return "—"
    if n < MIN_READABLE_N:
        return "n too small to read"
    low, high = wilson_interval(correct, n)
    return f"{low:.2%}–{high:.2%}"


def _wilson_cell(stats: Mapping[str, Any]) -> str:
    return _interval_text(stats["correct"], stats["decided"])


def _band_cell(stats: Mapping[str, Any]) -> str:
    if not stats["decided"]:
        return "—"
    if stats["decided"] < MIN_READABLE_N:
        return f"{stats['correct']}/{stats['decided']} (n too small to read)"
    return f"{stats['correct']}/{stats['decided']} ({stats['precision']:.0%})"


def _best_class_line(stats: Mapping[str, Any]) -> str:
    """Best-class precision over decided answers (review R2.2): unclear is not a miss."""
    return (f"Best-class suggestions heard as the recitation: **{stats['best_correct']} of "
            f"{stats['decided']} decided** ({stats['n']} answered; Wilson 95% "
            f"{_interval_text(stats['best_correct'], stats['decided'])}).")


def _stratum_views(worklist: Sequence[Mapping[str, Any]]) -> dict[str, list[dict]]:
    """Each stratum's rows as single-stratum views, so a shared row is scored in every one."""
    return {stratum: [{**row, "stratum": stratum} for row in worklist
                      if stratum in row["strata"]]
            for stratum in AUDIT_SIZES}


def _by_band(rows: Iterable[Mapping[str, Any]], d_max: float) -> dict[str, list[dict]]:
    bands: dict[str, list[dict]] = {band: [] for band in DURATION_BANDS}
    for row in rows:
        band = band_of(row["duration_s"], d_max)
        if band is not None:
            bands[band].append(dict(row))
    return bands


def _within_ayat(first: object, second: object, distance: int) -> bool:
    """Whether two ``surah:ayah`` labels are within ``distance`` ayat of one another."""
    try:
        surah_a, ayah_a = (int(part) for part in str(first).split(":"))
        surah_b, ayah_b = (int(part) for part in str(second).split(":"))
    except ValueError:
        return False
    return surah_a == surah_b and 0 < abs(ayah_a - ayah_b) <= distance


def read_ident_tree(population: str, *, ident_dir: Path = IDENT_DIR) -> list[dict]:
    """One committed identification tree, by population name.

    A population with splits is a directory of one file per split (``look/``); a single-file
    population is ``<name>.jsonl.gz`` (``train_sample``). S7 reuses this reader for its tranche
    manifest: one tree per population, read by name.
    """
    base = Path(ident_dir) / population
    paths = (sorted(base.glob("*.jsonl.gz")) if base.is_dir()
             else [Path(ident_dir) / f"{population}.jsonl.gz"])
    if not paths or not all(path.is_file() for path in paths):
        raise FileNotFoundError(f"no identification tree for {population} under {ident_dir}")
    return [row for path in paths for row in read_jsonl_gz(path)]


def _ident_rows(ident_dir: Path) -> list[dict]:
    """The two S6 populations' committed ident rows, each tagged with its population."""
    rows = [{**row, "population": "look"}
            for row in read_ident_tree("look", ident_dir=ident_dir)]
    rows += [{**row, "population": "train_sample"}
             for row in read_ident_tree("train_sample", ident_dir=ident_dir)]
    return rows


def _floor_frame(ident_dir: Path) -> dict[str, int]:
    """The frame the ``non_recitation`` stratum samples.

    Every decoded row that fails the floor is in the stratum (architect r22); those that also
    match their filename class are drawn in ``both`` too and scored in both, so the count here
    is disclosed next to the precision rather than removed from it.
    """
    failing = [row for row in _ident_rows(ident_dir)
               if row["decode_mode"] == WHOLE_CLIP and is_non_recitation(row)]
    return {"failing": len(failing),
            "both": sum(1 for row in failing if row.get("label_source") == BOTH)}


def _floor_prefix(ident_dir: Path) -> list[dict]:
    """The §9 sample: the first 25 floor-failing rows by the frozen salt's rank (r22).

    The draw and the fallback write the worklist's ``floor_sample`` marker; this derivation
    from the committed ident trees is what binds that marker to the frozen rule
    (``test_the_committed_worklist_marks_the_frozen_floor_sample``).
    """
    failing = [row for row in _ident_rows(ident_dir)
               if row["decode_mode"] == WHOLE_CLIP and is_non_recitation(row)]
    ranked = sorted(failing, key=lambda row: rank(SALT, "non_recitation", *row_key(row)))
    return ranked[:AUDIT_SIZES["non_recitation"]]


def _label_independent(ident_dir: Path) -> dict[str, float]:
    """r21 §2a/§2d: how the floor behaves on tlog, from the committed ident trees.

    The shortlist is label-free, so a best match landing on the labelled ayah is not caused by
    the label: at about 6,053 text classes that is not chance. Retention is the share of
    decoded clean look rows the floor keeps; the calibration's own retention comes from
    ``ident_calibration.json``.
    """
    decoded = [row for row in _ident_rows(ident_dir)
               if row["population"] == "look" and row["decode_mode"] == WHOLE_CLIP]
    clean = [row for row in decoded if row["split"] == "clean"]
    labelled = [row for row in clean if row.get("filename_label")]
    flagged = [row for row in labelled if is_non_recitation(row)]
    if not flagged:
        raise SystemExit("no flagged clean look row: the §2a disclosure would be empty")
    same = sum(1 for row in flagged if row.get("class_label") == row["filename_label"])
    near = sum(1 for row in flagged
               if _within_ayat(row.get("class_label"), row["filename_label"], 2))
    calibration = json.loads(Path(tlog_ident.CALIBRATION_PATH).read_text(encoding="utf-8"))
    return {"flagged": len(flagged), "same": same, "near": near,
            "retention": sum(1 for row in clean if row["id_floor_ok"] is True) / len(clean),
            "calibration_retention": calibration["look"]["retention"]}


def _require_verdicts(worklist: Sequence[Mapping[str, Any]],
                      verdicts: Mapping[str, Any], path: Path) -> None:
    """Refuse a document built from no verdicts, or from verdicts off another worklist."""
    if not verdicts:
        raise SystemExit(f"{path} has no verdicts: commit the owner's sitting before reporting")
    stray = sorted(set(verdicts) - {_item_key(row) for row in worklist})
    if stray:
        raise SystemExit(f"{path}: {len(stray)} verdicts name items that are not in the "
                         f"worklist ({', '.join(stray[:3])})")


def render_audit(*, worklist_path: Path = AUDIT_DIR / "worklist.jsonl",
                 verdicts_path: Path = AUDIT_DIR / "verdicts.jsonl",
                 ident_dir: Path = IDENT_DIR) -> str:
    """Render ``docs/tlog-audit.md`` from the committed worklist and verdicts.

    A render from zero verdicts would assert ``trigger fired: False`` from no evidence, so it
    refuses instead; a partly answered sitting still renders.
    """
    worklist = [dict(row) for row in read_jsonl(worklist_path)]
    if not worklist:
        raise SystemExit(f"{worklist_path} is empty: there is no audit to report")
    verdicts = read_verdicts(verdicts_path)
    _require_verdicts(worklist, verdicts, verdicts_path)
    d_max = float(decode_limit()["d_max_seconds"])
    database, _bars, _k = load_identifier()
    canonical = lambda label: tlog_ident.canonical_label(database, label)
    by_stratum = _stratum_views(worklist)
    # The floor table and §9 read the frozen sample only (architect r22): a fallback's
    # restricted-only extras are reported, never counted. An older worklist carries no marker
    # at all, which would silently count extras, so it is refused.
    floor_rows = [{**row, "stratum": "non_recitation"} for row in worklist
                  if row.get("floor_sample")]
    extras = [{**row, "stratum": "non_recitation"} for row in worklist
              if "non_recitation" in row["strata"] and not row.get("floor_sample")]
    if extras and not floor_rows:
        raise SystemExit(f"{worklist_path} marks no §9 sample row: the floor outcome cannot "
                         "tell the frozen sample from an earlier draw's extras")
    by_stratum["non_recitation"] = floor_rows
    lines = ["# tlog S6 audit (#191)", "",
             "Generated by `python -m tadabur.tlog_audit report` from",
             "`tools/tadabur/tlog/audit/{worklist,verdicts}.jsonl`. Never hand-edited.", "",
             f"The S6 amendment (`docs/tlog-s6-amendment.md`, SHA-256 "
             f"`{sha256_file(AMENDMENT_PATH)}`) was approved by the owner on "
             f"{AMENDMENT_APPROVAL['date']} ({AMENDMENT_APPROVAL['url']}).", "",
             f"Every cited commit resolves under `refs/pull/{PULL_REQUEST}/head` (PR "
             f"#{PULL_REQUEST}) after the squash merge.", "",
             f"{len(verdicts)} of {len(worklist)} drawn items answered. A Wilson 95% interval "
             f"is reported at any n; below {MIN_READABLE_N} decided items it reads "
             "`n too small to read` (amendment §4).", "",
             "## Verdicts", "",
             "| stratum | drawn | answered | decided | right | precision | Wilson 95% |",
             "|---|---|---|---|---|---|---|"]
    for stratum in AUDIT_SIZES:
        rows = by_stratum.get(stratum, [])
        stats = _precision(rows, verdicts, canonical)
        unreadable = bool(stats["decided"]) and stats["decided"] < MIN_READABLE_N
        precision = ("n too small to read" if unreadable else
                     "—" if stats["precision"] is None else f"{stats['precision']:.2%}")
        lines.append(f"| {stratum} | {len(rows)} | {stats['n']} | {stats['decided']} | "
                     f"{stats['correct']} | {precision} | {_wilson_cell(stats)} |")
    lines += ["", "## Precision by duration band", "",
              "| stratum | 0-20 s | 20-60 s | 60-d_max |", "|---|---|---|---|"]
    for stratum in AUDIT_SIZES:
        bands = _by_band(by_stratum.get(stratum, []), d_max)
        cells = [_band_cell(_precision(bands[band], verdicts, canonical))
                 for band in DURATION_BANDS]
        lines.append(f"| {stratum} | " + " | ".join(cells) + " |")
    both = _precision(by_stratum.get("both", []), verdicts, canonical)
    wrong = 1 - both["precision"] if both["precision"] is not None else None
    trigger = wrong is not None and wrong > OUTSIDE_ASR_LIMIT
    lines += ["", "## Outside-ASR trigger", "",
              f"- Residual wrong-ayah rate in `both`: "
              f"{'—' if wrong is None else f'{wrong:.2%}'} (limit "
              f"{OUTSIDE_ASR_LIMIT:.0%}).",
              f"- Trigger fired: **{trigger}**."
              + (" A follow-up issue is opened for an outside-family ASR cross-check."
                 if trigger else "")]
    drawn_floor = len(by_stratum["non_recitation"])
    floor = _precision(by_stratum["non_recitation"], verdicts, canonical)
    extras_stats = _precision(extras, verdicts, canonical)
    paused = floor["recitation"] >= FLOOR_PAUSE_AT
    lines += ["", "## The S5b §9 trigger", "",
              f"- Sample: the first {drawn_floor} rows by the frozen salt's rank among all "
              "decoded rows failing the floor, whatever other stratum they also belong to "
              f"(`{OVERLAP_NOTE_PATH.relative_to(REPO_ROOT).as_posix()}` §3).",
              f"- Audited `non_recitation` clips heard as real recitation: "
              f"**{floor['recitation']} of {drawn_floor} drawn** ({floor['n']} answered).",
              f"- Restricted-only extras kept from an earlier draw: {len(extras)} "
              f"({extras_stats['n']} answered, {extras_stats['recitation']} heard as real "
              "recitation). Reported, never counted in §9.",
              f"- Fallback not triggered: the full-frame redraw went live before any verdict "
              f"(`{FLOOR_REDRAW_COMMIT}`, PR #{PULL_REQUEST}, "
              f"`refs/pull/{PULL_REQUEST}/head`).",
              f"- Pre-stated before any verdict: {FLOOR_PAUSE_AT} or more suspends the 5% "
              "rule's consequence pending a dated floor revision the owner decides.",
              f"- **Suspended: {paused}.**"
              + (" The whole-pool decode and the `not non_recitation` filter do not start."
                 if paused else " They proceed as frozen, through the follow-up issue."),
              "- The follow-up issue for the whole-pool decode and the filter is drafted, not "
              "filed: it is blocked on the owner's decision and on this outcome.", ""]
    below = _precision(by_stratum["below-bar disagreement"], verdicts, canonical)
    below_precision = ("—" if below["precision"] is None
                       else f"{below['precision']:.2%}")
    lines += ["", "## The best class in the below-bar stratum (reported, never gating)", "",
              f"- {_best_class_line(below)}",
              f"- Filename-label precision in the same items: {below_precision}.", ""]
    silence = _precision(by_stratum.get("silence", []), verdicts, canonical)
    unclean = [row for row in by_stratum.get("silence", []) if row["split"] == "unclean"]
    lines += ["## The recitation floor in the field", "",
              f"- `non_recitation` items heard as real recitation: {floor['recitation']} of "
              f"{drawn_floor} drawn ({floor['n']} answered).",
              f"- silence items heard as real recitation: {silence['recitation']} of "
              f"{len(by_stratum.get('silence', []))} drawn; unclean alone {len(unclean)} "
              "drawn.", ""]
    meta_path = Path(ident_dir) / "train_sample.meta.json"
    if not meta_path.is_file():
        raise SystemExit(f"{meta_path} is absent: the 5% rule has not been computed")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    rule = meta["five_percent_rule"]
    lines += ["## The 5% rule", "",
              f"- sample n = {rule['n']}, non_recitation = {rule['non_recitation']}, "
              f"Wilson 95% upper = {rule['wilson'][1]:.2%}, triggered = "
              f"**{rule['triggered']}**.", "",
              "| band | rows | non_recitation |", "|---|---|---|"]
    for band in DURATION_BANDS:
        counts = meta["bands"][band]
        lines.append(f"| {band} | {counts['rows']} | {counts['non_recitation']} |")
    lines.append("")
    if not OVERLAP_META_PATH.is_file():
        raise SystemExit(f"{OVERLAP_META_PATH} is absent: the S4 disclosure cannot be skipped")
    overlap = json.loads(OVERLAP_META_PATH.read_text(encoding="utf-8"))
    words = overlap["controls"]["stop_word_matches"]["accepted"]
    survivors = overlap["duplicates"]["survivor_unusable"]
    lines += ["## Disclosure: the S4 exclusions (not audited here)", "",
              f"- {words['near_duplicate']['share']:.2%} of accepted near-duplicate matches are "
              f"dominated by stop-words; {words['all']['share']:.2%} combined, above the 1% "
              "warning.",
              f"- {survivors['eval']['total']} eval-role and {survivors['train']['total']} "
              "train usable rows were lost as `survivor_unusable`.",
              "- No audited item: those rows are excluded.", ""]
    field = _label_independent(Path(ident_dir))
    frame = _floor_frame(Path(ident_dir))
    lines += ["## Disclosure: the floor's calibration population", "",
              f"- The floor and its 0.30 decoy margin were calibrated on Tarteel v1, whose "
              "clips were filtered upstream (`cer_wav <= 0.299`, completeness >= 0.85); tlog "
              f"is unfiltered. Tarteel look retention was "
              f"{field['calibration_retention']:.2%}; tlog clean look retention is "
              f"{field['retention']:.2%}.",
              f"- The shortlist is label-free, so a best match on the labelled ayah is not "
              f"driven by the label: of {field['flagged']} flagged clean look rows, "
              f"{field['same']} ({field['same'] / field['flagged']:.1%}) match exactly the "
              f"labelled ayah and {field['near']} ({field['near'] / field['flagged']:.1%}) a "
              "neighbour within two ayat. Many flagged rows are recitation the teacher decodes "
              "poorly, not non-recitation.",
              f"- Sampling frame of the `non_recitation` stratum: all {frame['failing']} decoded "
              "rows that fail the floor. "
              f"{frame['both']} of them also match their filename class and are drawn in `both` "
              "as well; per "
              f"`{OVERLAP_NOTE_PATH.relative_to(REPO_ROOT).as_posix()}` each such row is heard once "
              "and scored in both strata, so the precision above covers the whole flag rather "
              "than a subset of it.", ""]
    lines += ["## Served order", "",
              "- The page serves the committed worklist in its written (shard) order rather "
              "than the amendment §5 salted shuffle: the worklist was frozen once verdicts "
              "began, and the order carries no stratum information.", ""]
    return "\n".join(lines).rstrip() + "\n"


# --- CLI ---------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("audit", help="draw the stratified audit worklist")
    stage_ = commands.add_parser("stage", help="stage the drawn clips for listening")
    stage_.add_argument("--audio-dir", type=Path, required=True)
    stage_.add_argument("--cache-dir", type=Path, required=True)
    stage_.add_argument("--draw", type=Path, default=AUDIT_DIR / "draw.jsonl",
                        help="the draw to stage (default: audit/draw.jsonl)")
    report_ = commands.add_parser("report", help="render docs/tlog-audit.md")
    report_.add_argument("--audio-dir", type=Path, default=None,
                         help="the session-only staged audio to delete after the report")
    args = parser.parse_args()
    if args.command == "audit":
        draw = audit_draw(audit_candidates())
        write_jsonl(draw.items, AUDIT_DIR / "draw.jsonl", sort_keys=True)
        print(json.dumps(draw.strata, indent=1, sort_keys=True), flush=True)
    elif args.command == "stage":
        draw = AuditDraw(items=tuple(read_jsonl(args.draw)), strata={})
        stage(draw, audio_dir=args.audio_dir, cache_dir=args.cache_dir)
    else:
        write_text_atomically(DOC_PATH, render_audit())
        if args.audio_dir is not None:
            shutil.rmtree(args.audio_dir, ignore_errors=True)
            print(f"deleted the staged audio in {args.audio_dir}", flush=True)
        print(f"wrote {DOC_PATH}", flush=True)


if __name__ == "__main__":
    main()

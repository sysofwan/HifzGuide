"""Gate P1 of the follow-along audit (#306): Muraja's own device logs and the owner's sessions,
replayed at the frozen pin and at the audit's commit.

* **Muraja's logs.** Every log in ``tools/test_logs`` at :data:`CURRENT_COMMIT` replays under the
  shipped default and reproduces the ``baseline.json`` that ships with it (the final position and
  every word's quality), exactly.
* **The owner's sessions** (app 1.0.27, recorded at :data:`MURAJA_COMMIT`). Each reproduces every
  ``Grade end word`` decision the device logged, in the session's own mode, with #153's lengthened
  previews. At :data:`CURRENT_COMMIT` (app 1.0.28) a decision may differ; each differing word is
  listed with the 1.0.28 change that explains it (:data:`CHANGES`), from a mechanical rule per
  change (:func:`explain`). A difference no rule explains stops the report.

The committed report is ``tools/device_logs/parity_report.json`` and the note is
``docs/followalong-audit-parity.md``. The owner's exports are private: the report and the note
carry positions and counts only.

Usage (from ``tools/``, with the harness built at both commits)::

    python -m training.followalong_parity ~/Downloads/muraja-bug-report\\ {69,70,71}.txt
"""

from __future__ import annotations

import argparse
import json
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path

from training import device_exports as exports
from training import device_logs as device
from training import word_flags
from training.muraja_policy import CURRENT_COMMIT, MURAJA_COMMIT, SHIPPED_DEFAULT, MurajaText, Scoring
from training.tashkeel_eval import write_text_atomically

DEVICE_LOGS_DIR = Path(__file__).parent.parent / "device_logs"
REPORT_PATH = DEVICE_LOGS_DIR / "parity_report.json"
DOC_PATH = Path(__file__).parent.parent.parent / "docs" / "followalong-audit-parity.md"
#: The 1.0.28 changes (Muraja ``99c326f..028bc51``) a device decision may differ by.
CHANGES = ("madd_alif", "tanween_at_waqf", "stop_pass", "error_indexing")
#: Muraja's ``tools/test_logs`` at the audit's commit, each with its ``baseline.json``.
LOG_COUNT = 74
_GRADE = re.compile(r"Grade end word (\d+:\d+:\d+): ")
_FIELD = {"cov": r"cov=([\d.]+)", "gap": r"gap=(\d+)"}
#: The tanween marks Muraja's ``wordHasTanween`` reads (``PhonemeIndex.swift:94-103``):
#: U+064B-U+064D and U+08F0-U+08F2.
_TANWEEN = frozenset(range(0x064B, 0x064E)) | frozenset(range(0x08F0, 0x08F3))


def grade_lines(replay) -> list[str]:
    """A replay's decisions, in order: the word scorer's ``Grade end word`` lines."""
    return [line for line in replay.grade_lines if line.startswith("Grade end word")]


def by_word(lines: Sequence[str]) -> dict[str, list[str]]:
    """The decisions per word position."""
    out: dict[str, list[str]] = {}
    for line in lines:
        if (match := _GRADE.match(line)) is not None:
            out.setdefault(match[1], []).append(line)
    return out


def _field(lines: Sequence[str], name: str) -> list[float]:
    return [float(match[1]) for line in lines if (match := re.search(_FIELD[name], line)) is not None]


def _without(lines: Sequence[str], name: str) -> list[str]:
    """The lines with one counted field removed: equality then says every other field held."""
    return [re.sub(rf" {name}=\S+", "", line) for line in lines]


def _alifs(word: str) -> int:
    """The madd alifs of a phoneme word: in Muraja's reference the alif is only ever a madd."""
    return sum(unit.startswith("ا") for unit in word_flags.units(word))


def _unit_rows(word: str, rows: Sequence[str]) -> list[int]:
    """The ``phoneme_groups`` row of each unit Muraja's scorer numbers, in order (empty when the
    rows and the word do not line up, and no index is then read)."""
    mapping = [row for row, text in enumerate(rows) for _ in word_flags.units(text)]
    return mapping if len(mapping) == len(word_flags.units(word)) else []


def _key(error: Mapping) -> tuple:
    kind = error["kind"]
    return kind["type"], kind.get("expected"), kind.get("heard")


def _row_indexed(pin_errors: Sequence[Mapping], current_errors: Sequence[Mapping],
                 word: str, rows: Sequence[str]) -> bool:
    """The pin's errors, each re-read at its ``phoneme_groups`` row and one per row, are the
    current errors: the index moved and a duplicate collapsed (Muraja ``e0b8b18``)."""
    mapping = _unit_rows(word, rows)
    if not mapping:
        return False
    collapsed = []
    for row in range(len(rows)):
        first = next((error for error in pin_errors
                      if error["group_index"] < len(mapping) and mapping[error["group_index"]] == row), None)
        if first is not None:
            collapsed.append((row, _key(first)))
    current = [(error["group_index"], _key(error)) for error in current_errors]
    return collapsed == current and collapsed != [(error["group_index"], _key(error)) for error in pin_errors]


def explain(
    session: str, word: str, pin: Mapping | None, current: Mapping | None,
    pin_lines: Sequence[str], current_lines: Sequence[str], text: MurajaText,
) -> str:
    """The 1.0.28 change that explains why ``word``'s decision differs, by a mechanical rule per
    change; a difference none of them explains stops the report."""
    surah, ayah, index = device.position(word)
    phonemes = text.phonemes(surah, ayah).split(" ")[index - 1]
    rows = text.word_rows(surah, ayah, index)
    pin_gap = max(_field(pin_lines, "gap"), default=0)
    current_gap = max(_field(current_lines, "gap"), default=0)
    dropped = pin_gap - current_gap
    last_row = word_flags.units(rows[-1]) if rows else []
    # Tried in this order: the stop pass only says nothing else moved, so a rule that names the
    # moved field goes first and keeps its word even when the stop pass could also claim it.
    matched = {
        # c4f3a98: a gap on an alif is a shortened madd, no deletion error and no gate trip
        "madd_alif": (dropped > 0 and _alifs(phonemes) >= dropped
                      and _without(pin_lines, "gap") == _without(current_lines, "gap")),
        # b052c26: the reference trims a trailing nun of a tanween at a pause
        "tanween_at_waqf": (max(_field(current_lines, "cov"), default=0) > max(_field(pin_lines, "cov"), default=0)
                            and bool(last_row) and last_row[0][0] in "نں"
                            and any(ord(char) in _TANWEEN for char in text.printed(surah, ayah, index))),
        # e0b8b18: WordError.groupIndex is the phoneme_groups row, one error per row
        "error_indexing": _row_indexed((pin or {}).get("errors", ()), (current or {}).get("errors", ()),
                                       phonemes, rows),
        # f06530c, baf4726: the stop pass regrades the words it reaches after the last check
        "stop_pass": list(pin_lines) == list(current_lines),
    }
    for name, applies in matched.items():
        if applies:
            return name
    raise SystemExit(
        f"{session}: {word}: no named 1.0.28 change explains the difference: "
        f"pin {pin} / current {current}")


def session_differences(session: str, pin_replay, current_replay, text: MurajaText) -> list[dict]:
    """Every word whose decisions or kept status differ between the two commits, each with the
    change that explains it."""
    pin_lines = by_word(grade_lines(pin_replay))
    current_lines = by_word(grade_lines(current_replay))
    pin_words = dict(pin_replay.words)
    current_words = dict(current_replay.words)
    if pin_replay.final_position != current_replay.final_position:
        raise SystemExit(f"{session}: the final position moved from {pin_replay.final_position} "
                         f"to {current_replay.final_position}; no named change explains that")
    differences = []
    for word in sorted(set(pin_lines) | set(current_lines) | set(pin_words) | set(current_words),
                       key=device.position):
        if pin_lines.get(word) == current_lines.get(word) and pin_words.get(word) == current_words.get(word):
            continue
        differences.append({
            "word": word,
            "explained_by": explain(session, word, pin_words.get(word), current_words.get(word),
                                    pin_lines.get(word, ()), current_lines.get(word, ()), text),
        })
    return differences


def log_parity(harness) -> list[dict]:
    """Every log of the audit's commit against the baseline that ships with it, in order."""
    directory = device.log_dir(CURRENT_COMMIT)
    logs = sorted(path.stem for path in directory.glob("*.log"))
    baselines = sorted(path.name for path in directory.glob("*.baseline.json"))
    if len(logs) != LOG_COUNT or len(baselines) != LOG_COUNT:
        raise SystemExit(f"{directory} holds {len(logs)} logs and {len(baselines)} baselines, not {LOG_COUNT}")
    _, replays = device.replay_logs(
        harness, {log: directory / f"{log}.log" for log in logs}, [SHIPPED_DEFAULT])
    return [{
        "log": log,
        "differences": device.baseline_differences(
            replays[log][SHIPPED_DEFAULT.engine_name],
            json.loads((directory / f"{log}.baseline.json").read_text(encoding="utf-8"))),
    } for log in logs]


def owner_parity(pin, current, paths: Sequence[Path]) -> list[dict]:
    """The owner's sessions at the pin (exact) and at the audit's commit (every difference
    explained), each in the mode the device ran."""
    text = MurajaText(pin.quran_db)
    notes = exports.load_annotations(text)
    _, sessions = exports.read_exports(paths, notes)
    targets = [session for session in sessions
               if session.whole and exports.MODEL_OF.get(session.model) == exports.H448
               and session.app_version == exports.APP_VERSION]
    lengthened = {session.name: device.lengthened_previews(session.text)[0] for session in targets}
    with tempfile.TemporaryDirectory() as tmp:
        written = {name: Path(tmp) / f"{name}.log" for name in lengthened}
        for name, log in lengthened.items():
            written[name].write_text(log, encoding="utf-8")
        runs = {name: device.replay_logs(harness, written, exports.MODE_SCORINGS)[1]
                for name, harness in (("pin", pin), ("current", current))}
        rows = []
        for session in targets:
            device_lines = exports.grade_lines(session.text)

            def lines(run, scoring: Scoring) -> list[str]:
                return grade_lines(run[session.name][scoring.engine_name])

            mode = max(exports.MODE_SCORINGS,
                       key=lambda m: exports.parity(device_lines, lines(runs["pin"], m))["matched"]).mode
            scoring = next(m for m in exports.MODE_SCORINGS if m.mode == mode)
            pin_replay = runs["pin"][session.name][scoring.engine_name]
            current_replay = runs["current"][session.name][scoring.engine_name]
            rows.append({
                "session": session.name,
                "mode": mode,
                "pin_exact": exports.parity(device_lines, grade_lines(pin_replay))["exact"],
                "current_differences": session_differences(session.name, pin_replay, current_replay, text),
            })
    return rows


def build_report(paths: Sequence[Path]) -> dict:
    pin, current = device.require(MURAJA_COMMIT), device.require(CURRENT_COMMIT)
    logs = log_parity(current)
    if broken := [entry["log"] for entry in logs if entry["differences"]]:
        raise SystemExit(f"{current}'s replay does not reproduce its own baselines: {broken}")
    sessions = owner_parity(pin, current, paths)
    if inexact := [row["session"] for row in sessions if not row["pin_exact"]]:
        raise SystemExit(f"the pin's replay does not reproduce the device's decisions: {inexact}")
    return {"muraja": {"pin": MURAJA_COMMIT, "current": CURRENT_COMMIT},
            "logs": logs, "owner_sessions": sessions}


def main() -> None:
    from training.followalong_parity_doc import render

    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("exports", nargs="+", type=Path, help="the owner's bug-report exports")
    report = build_report(parser.parse_args().exports)
    write_text_atomically(REPORT_PATH, json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    write_text_atomically(DOC_PATH, render(report))
    print(f"Wrote {REPORT_PATH} and {DOC_PATH}")


if __name__ == "__main__":
    main()

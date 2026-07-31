"""What one checkpoint did at each already-labelled tashkeel site — no listening required.

This is the piece that decouples the audit from the training run (#61). A verdict from
:mod:`tadabur.tashkeel_audit_ui` records *what the reciter said* at a position, which is a
fact about the audio and outlives every checkpoint. What changes per fine-tune is only
whether that checkpoint reproduced the vowel — and that is a decode, not a listening
session.

So the expensive half of the audit is paid once, in advance, against the static worklist
:func:`training.tashkeel_worklist.static_sites` mines off the frozen base model. Every
candidate thereafter runs this module (minutes of GPU) instead of another audit sitting
(hours of a human), and :mod:`tadabur.tashkeel_acceptance` joins the two.

Only the windows the worklist actually references are decoded, so a 450-site worklist costs
a few hundred window decodes rather than a pass over the whole split.

The outcome vocabulary is :mod:`training.tashkeel_eval`'s, unchanged: ``matched`` means the
checkpoint put the reference vowel on the reference carrier, and everything in
``FAILED_OUTCOMES`` means it did not. A site the checkpoint's alignment never reaches is
recorded as ``unanchored`` by the same rules the aggregate report uses, so a number from
here and a number from ``tashkeel_eval`` cannot disagree about what "failed" means.

Runs on Linux + CUDA (see ``tools/README.md``).

Usage::

    python -m training.tashkeel_outcomes \\
        --worklist  audit_run/seg_v21/tashkeel_static.jsonl \\
        --labels    audit_run/seg_v21/windowed_labels_v2.jsonl \\
        --audio-dir audit_run/clips_v2 \\
        --model     audit_run/seg_v21/rung4/merged \\
        --out       audit_run/seg_v21/tashkeel_outcomes_rung4.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from training.tashkeel_eval import _decode_windows, _load_windows, vowel_sites
from training.tashkeel_worklist import TashkeelSite, read_worklist, site_id


@dataclass(frozen=True)
class SiteOutcome:
    """One checkpoint's result at one audited position."""

    site_id: str
    outcome: str
    decoded_vowel: str | None


OUTCOME_FIELDS = tuple(f.name for f in fields(SiteOutcome))


def _windows_for(worklist: list[TashkeelSite]) -> set[tuple[str, int]]:
    """The ``(clip, window)`` pairs the worklist touches — the only ones worth decoding."""
    return {(row.clip_audio_filename, row.window_index) for row in worklist}


def outcomes_for_window(
    reference: str, decode: str, label, wanted: set[str]
) -> list[SiteOutcome]:
    """Classify this checkpoint's decode at every wanted site in one window.

    Sites are matched on :func:`training.tashkeel_worklist.site_id`, i.e. on the reference
    index, which is the coordinate every alignment shares. ``wanted`` filters to the sites
    actually under audit so a window contributes nothing extra.
    """
    found: list[SiteOutcome] = []
    for site in vowel_sites(decode, reference):
        if site.reference_vowel is None or site.reference_index is None:
            continue
        identifier = site_id(label.clip_audio_filename, label.window_index, site.reference_index)
        if identifier in wanted:
            found.append(
                SiteOutcome(
                    site_id=identifier,
                    outcome=site.outcome,
                    decoded_vowel=site.decoded_vowel,
                )
            )
    return found


def write_outcomes(path: Path, rows: list[SiteOutcome]) -> None:
    """One JSON object per line, ordered by site id so two runs diff cleanly."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in sorted(rows, key=lambda r: r.site_id):
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def read_outcomes(path: Path) -> dict[str, SiteOutcome]:
    """Read outcomes keyed by site id, rejecting anything that is not this schema.

    A silently-tolerated stray field here would be a checkpoint's result being read into the
    wrong column of a comparison, so the schema is checked rather than trusted.
    """
    rows: dict[str, SiteOutcome] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = set(OUTCOME_FIELDS) - set(record)
        unknown = set(record) - set(OUTCOME_FIELDS)
        if missing or unknown:
            raise ValueError(
                f"{path}:{number} does not match the outcome schema "
                f"(missing: {sorted(missing)}, unknown: {sorted(unknown)})."
            )
        rows[record["site_id"]] = SiteOutcome(**record)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, required=True,
                        help="the audited worklist (training.tashkeel_worklist).")
    parser.add_argument("--labels", type=Path, required=True,
                        help="windowed CTC labels JSONL the worklist was mined from.")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="staged 16 kHz clip directory.")
    parser.add_argument("--model", required=True,
                        help="checkpoint to score at the audited sites.")
    parser.add_argument("--split", default="val",
                        help="label split the worklist was mined from.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True, help="outcomes JSONL.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    worklist = read_worklist(args.worklist)
    wanted = {row.site_id for row in worklist}
    windows = _windows_for(worklist)

    labels = [
        label
        for label in _load_windows(args.labels, args.split, None)
        if (label.clip_audio_filename, label.window_index) in windows
    ]
    if len(labels) < len(windows):
        raise ValueError(
            f"{args.labels} '{args.split}' is missing {len(windows) - len(labels)} of the "
            f"{len(windows)} windows the worklist references — the worklist was mined from "
            "a different label file or split, and scoring it here would silently drop sites."
        )
    print(
        f"Scoring {args.model} at {len(wanted)} sites across {len(labels)} windows.",
        flush=True,
    )

    decodes = _decode_windows(args.model, labels, args.audio_dir, args.batch_size, args.device)

    rows: list[SiteOutcome] = []
    for label, decode in zip(labels, decodes):
        rows.extend(outcomes_for_window(label.phoneme_label, decode, label, wanted))

    missing = wanted - {row.site_id for row in rows}
    if missing:
        raise ValueError(
            f"{len(missing)} audited sites got no outcome from {args.model}. Every reference "
            "vowel is classified by exactly one site, so this means the worklist and the "
            "labels disagree about a window's reference — not that the model skipped them."
        )

    write_outcomes(args.out, rows)
    print(f"Wrote {len(rows)} outcomes to {args.out}")


if __name__ == "__main__":
    main()

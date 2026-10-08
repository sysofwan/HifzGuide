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

The file opens with a ``{"decode": ...}`` header holding the
:class:`training.decoding.DecodeFingerprint` of the decode behind it;
:mod:`tadabur.tashkeel_acceptance` refuses outcomes whose fingerprint does not match the
worklist's frozen base decode, and refuses a file with none.

``--model`` is any reference :class:`training.decoding.Decoder` loads -- the base teacher's
hub id, a merged model directory, or a distilled student's checkpoint -- so the base model
and ``h448`` are scored at the same sites the same way.

Runs on Linux + CUDA (see ``tools/README.md``).

Usage::

    python -m training.tashkeel_outcomes \\
        --worklist  audit_run/seg_v21/tashkeel_static.jsonl \\
        --labels    audit_run/seg_v21/windowed_labels_v2.jsonl \\
        --audio-dir audit_run/clips_v2 \\
        --model     runs/h448_stream/checkpoint.pt \\
        --out       audit_run/seg_v21/tashkeel_outcomes_rung4.jsonl
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from training.decoding import DecodeFingerprint
from training.tashkeel_eval import (
    VOWEL_OUTCOMES,
    _decode_windows,
    _load_windows,
    vowel_sites,
    write_text_atomically,
)
from training.tashkeel_worklist import TashkeelSite, read_worklist, site_id


@dataclass(frozen=True)
class SiteOutcome:
    """One checkpoint's result at one audited position."""

    site_id: str
    outcome: str
    decoded_vowel: str | None


OUTCOME_FIELDS = tuple(f.name for f in fields(SiteOutcome))


def _windows_for(worklist: list[TashkeelSite]) -> dict[tuple[str, int], TashkeelSite]:
    """The ``(clip, window)`` pairs the worklist touches, each with one representative site.

    The representative carries the reference the listener was actually shown, which
    :func:`check_references` needs: ``site_id`` hashes only clip, window and reference index,
    so it cannot by itself detect that a *different* label file put a different phoneme string
    at that coordinate.
    """
    return {(row.clip_audio_filename, row.window_index): row for row in worklist}


def check_references(labels, windows: dict[tuple[str, int], TashkeelSite]) -> None:
    """Refuse to score if a window's reference is not the one the sites were mined from.

    ``site_id`` is deliberately model-free and coordinate-only, which is what lets a verdict
    outlive the checkpoint that surfaced it. The cost is that it is also *reference*-free: run
    this against a label file whose window 0 of ``clip.wav`` holds a different phoneme string
    and every id still matches, so outcomes computed against one reference would be joined to
    verdicts collected against another. Nothing downstream could detect it — the counts would
    simply be wrong.

    The span is checked too. Same coordinate, different windowing, is the same failure with a
    different cause: the listener graded audio the decode never saw.
    """
    for label in labels:
        site = windows[(label.clip_audio_filename, label.window_index)]
        if label.phoneme_label != site.reference:
            raise ValueError(
                f"{label.clip_audio_filename} window {label.window_index} has a different "
                "reference in the labels than the worklist recorded. The sites were "
                "adjudicated against the worklist's reference, so scoring against this one "
                "would join a decode of one text to verdicts about another."
            )
        if (label.start_sample, label.num_samples) != (site.start_sample, site.num_samples):
            raise ValueError(
                f"{label.clip_audio_filename} window {label.window_index} spans "
                f"({label.start_sample}, {label.num_samples}) in the labels but "
                f"({site.start_sample}, {site.num_samples}) in the worklist — the listener "
                "graded a different span of audio than this decode would cover."
            )


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


def write_outcomes(path: Path, rows: list[SiteOutcome], decode: DecodeFingerprint) -> None:
    """A ``{"decode": ...}`` header line, then one outcome per line ordered by site id.

    The header is the decode's fingerprint: these outcomes are compared against a worklist's
    frozen base outcomes, which is only sound under the same decode settings. The whole file
    is built before anything is written, and swapped in atomically.
    """
    lines = [json.dumps({"decode": decode.as_dict()}, ensure_ascii=False)] + [
        json.dumps(asdict(row), ensure_ascii=False)
        for row in sorted(rows, key=lambda r: r.site_id)
    ]
    write_text_atomically(path, "".join(line + "\n" for line in lines))


def read_outcomes(path: Path) -> tuple[DecodeFingerprint, dict[str, SiteOutcome]]:
    """Read the decode fingerprint and the outcomes keyed by site id, rejecting anything else.

    A file with no fingerprint header predates it (a legacy decode) and is refused: nothing
    can say whether it is comparable to the worklist it would be joined to.

    A silently-tolerated stray field here would be a checkpoint's result being read into the
    wrong column of a comparison, so the schema is checked rather than trusted.

    Outcome *values* are checked against :data:`VOWEL_OUTCOMES` too, because
    :func:`tadabur.tashkeel_acceptance._failed` asks whether an outcome is in the failed set —
    a denylist. Under a denylist a typo (``"omited"``) reads as the checkpoint having got the
    position right, quietly shrinking its false-rejection estimate, which is the one direction
    an audit of over-strictness must never fail in.

    Duplicate site ids are rejected rather than last-wins. Unlike an adjudications file, where
    a listener revising a verdict is expected, two outcomes for one site means two decodes were
    concatenated and there is no way to tell which checkpoint the survivor came from.
    """
    lines = [
        (number, line)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1)
        if line.strip()
    ]
    header = json.loads(lines[0][1]) if lines else {}
    decode = DecodeFingerprint.from_dict(
        header.get("decode") if set(header) == {"decode"} else None, str(path)
    )
    rows: dict[str, SiteOutcome] = {}
    for number, line in lines[1:]:
        record = json.loads(line)
        missing = set(OUTCOME_FIELDS) - set(record)
        unknown = set(record) - set(OUTCOME_FIELDS)
        if missing or unknown:
            raise ValueError(
                f"{path}:{number} does not match the outcome schema "
                f"(missing: {sorted(missing)}, unknown: {sorted(unknown)})."
            )
        if record["outcome"] not in VOWEL_OUTCOMES:
            raise ValueError(
                f"{path}:{number} has outcome {record['outcome']!r}, which is not one of "
                f"{sorted(VOWEL_OUTCOMES)}. Anything unrecognised would score as the "
                "checkpoint having got this position right."
            )
        if record["site_id"] in rows:
            raise ValueError(
                f"{path}:{number} repeats site {record['site_id']}. Two outcomes for one site "
                "means two decodes were concatenated; which checkpoint won is not recoverable."
            )
        rows[record["site_id"]] = SiteOutcome(**record)
    return decode, rows


def build_parser() -> argparse.ArgumentParser:
    from training.decoding import MODEL_REF_HELP, add_weights_dtype_argument

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worklist", type=Path, required=True,
                        help="the audited worklist (training.tashkeel_worklist).")
    parser.add_argument("--labels", type=Path, required=True,
                        help="windowed CTC labels JSONL the worklist was mined from.")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="staged 16 kHz clip directory.")
    parser.add_argument("--model", required=True,
                        help=f"model to score at the audited sites: {MODEL_REF_HELP}.")
    parser.add_argument("--split", default="val",
                        help="label split the worklist was mined from.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    add_weights_dtype_argument(parser)
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
    seen = {(label.clip_audio_filename, label.window_index) for label in labels}
    if len(seen) < len(windows):
        raise ValueError(
            f"{args.labels} '{args.split}' is missing {len(windows) - len(seen)} of the "
            f"{len(windows)} windows the worklist references — the worklist was mined from "
            "a different label file or split, and scoring it here would silently drop sites."
        )
    if len(labels) != len(seen):
        raise ValueError(
            f"{args.labels} '{args.split}' has {len(labels) - len(seen)} duplicate "
            "(clip, window) rows among the windows this worklist references. Decoding one "
            "coordinate twice would emit two outcomes for the same site."
        )
    check_references(labels, windows)
    print(
        f"Scoring {args.model} at {len(wanted)} sites across {len(labels)} windows.",
        flush=True,
    )

    decodes, fingerprint = _decode_windows(
        args.model, labels, args.audio_dir, args.batch_size, args.device, args.weights_dtype
    )

    rows: list[SiteOutcome] = []
    for label, window_decode in zip(labels, decodes):
        rows.extend(outcomes_for_window(label.phoneme_label, window_decode, label, wanted))

    missing = wanted - {row.site_id for row in rows}
    if missing:
        raise ValueError(
            f"{len(missing)} audited sites got no outcome from {args.model}. Every reference "
            "vowel is classified by exactly one site, so this means the worklist and the "
            "labels disagree about a window's reference — not that the model skipped them."
        )

    write_outcomes(args.out, rows, fingerprint)
    print(f"Wrote {len(rows)} outcomes to {args.out}")


if __name__ == "__main__":
    main()

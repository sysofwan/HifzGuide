"""The random listening audit that replaced ADR-0016's full adjudication pass.

Decision 10 originally required **every** corpus clip to be heard, because non-Hafs
recitation is orthogonal to ``match_ratio`` — :func:`~tadabur.normalization.normalize_phonemes`
deletes short vowels, so a wholly non-Hafs reading can score a perfect ratio — and no
automated screen exists. The amendment drops that, for a reason in the decision's own
second half: alignment runs on the *normalized* string and tashkeel is detected in a
separate expansion pass, so a **vowel-only** divergence cannot move the cursor, and the
grader sends it to ``.tashkeelError`` rather than ``.wrong`` or ``.skipped``. Nothing
this corpus measures — cycles-to-resync, falsely-skipped words — can see it.

That argument holds only while vowel-only divergence is the dominant mode. A
**consonantal** non-Hafs reading does perturb the alignment, and therefore the cursor.
The evidence for vowel-only dominance is one instance in 31 adjudicated clips, which is
not evidence. So the pass this module samples is not a screen — it does not decide which
clips ship — it is the **test of that assumption**, and it therefore records *how* a
non-Hafs reading diverges, not only that one did.

Sampling is off the staged bundle rather than the reject sink, so what is heard is what
Muraja replays: post-re-cut audio, at the boundaries the corpus asserts. It is seeded and
sorted, so the same bundle and seed draw the same clips — a re-run tops the sample up
rather than redrawing it.

Listening is a human step. This module stages the worklist and the audio for it, and
summarises the verdicts once they exist.

Usage:
  python -m tadabur.reread_audit --bundle corpus_run/scenario --out corpus_run/audit
    [--size 50] [--seed 0]
  python -m tadabur.reread_audit --summary
    [--verdicts eval_fixtures/reject_reread_verdicts.jsonl]
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path

from .scenario import read_scenario_records

DEFAULT_SIZE = 50
DEFAULT_SEED = 0
DEFAULT_VERDICTS_PATH = Path(__file__).parent / "eval_fixtures" / "reject_reread_verdicts.jsonl"

# The two verdicts a listener returns. ``nonhafs`` covers any reading that is not the
# Hafs 'an 'Asim transmission the reference text is in; everything else — including a
# clip carrying bleed or an unfinished ayah, which are separately detected and recorded
# — is ``clean`` for this pass's purposes.
VERDICT_CLEAN = "clean"
VERDICT_NONHAFS = "nonhafs"

# How a non-Hafs reading diverges, which is the whole point of the pass.
# ``vowel_only`` is inert to alignment (normalization deletes short vowels);
# ``consonantal`` is not, and any material rate of it reopens decision 10.
DIVERGENCE_VOWEL_ONLY = "vowel_only"
DIVERGENCE_CONSONANTAL = "consonantal"
DIVERGENCE_MODES = (DIVERGENCE_VOWEL_ONLY, DIVERGENCE_CONSONANTAL)


def sample_clip_ids(clip_ids: list[str], size: int, seed: int) -> list[str]:
    """``size`` clip ids drawn from ``clip_ids`` reproducibly.

    Sorted before sampling so the draw depends on the *set* of staged clips and the seed,
    not on the order the bundle happened to be written in. A ``size`` at or above the
    population returns all of it — a sample of everything is the whole corpus, not an
    error.
    """
    population = sorted(set(clip_ids))
    if size >= len(population):
        return population
    return sorted(random.Random(seed).sample(population, size))


def summarise_verdicts(rows: list[dict]) -> dict:
    """Counts behind the audit's one claim: how non-Hafs readings diverge.

    ``divergence_modes`` counts only the ``nonhafs`` rows, since a clean reading has no
    divergence to classify. A ``nonhafs`` row with no mode recorded is counted under
    ``unclassified`` rather than assumed vowel-only — assuming it is the inertness
    argument's conclusion, and this pass exists to test that, not to restate it.
    """
    verdicts = Counter(row.get("verdict", "") for row in rows)
    modes = Counter(
        row.get("divergence") or "unclassified"
        for row in rows
        if row.get("verdict") == VERDICT_NONHAFS
    )
    return {
        "judged": len(rows),
        "verdicts": dict(sorted(verdicts.items())),
        "nonhafs_rate": round(verdicts[VERDICT_NONHAFS] / len(rows), 4) if rows else 0.0,
        "divergence_modes": dict(sorted(modes.items())),
    }


def read_verdicts(path: Path) -> list[dict]:
    """Every verdict row in ``path``, skipping blanks and ``#`` comments."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                rows.append(json.loads(line))
    return rows


def _stage(bundle: Path, out: Path, size: int, seed: int) -> dict:
    records = {r.clip_id: r for r in read_scenario_records(bundle / "scenario.jsonl")}
    chosen = sample_clip_ids(list(records), size, seed)
    judged = {row.get("clip_id") for row in read_verdicts(DEFAULT_VERDICTS_PATH)} if DEFAULT_VERDICTS_PATH.exists() else set()

    audio_dir = out / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)
    with (out / "worklist.jsonl").open("w", encoding="utf-8") as f:
        for clip_id in chosen:
            record = records[clip_id]
            shutil.copyfile(bundle / record.audio, audio_dir / f"{clip_id}.wav")
            f.write(
                json.dumps(
                    {
                        "clip_id": clip_id,
                        "audio_ref": f"{clip_id}.wav",
                        "surah_ayah": record.surah_ayah,
                        "reciter_id": record.reciter_id,
                        "duration_s": round(record.duration_s, 3),
                        "match_ratio": round(record.match_ratio, 4),
                        "max_insertion_run": record.max_insertion_run,
                        "recut_applied": record.recut_applied,
                        "uncovered_tail": record.uncovered_tail,
                        "already_judged": clip_id in judged,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return {
        "staged_clips": len(records),
        "sampled": len(chosen),
        "already_judged": sum(1 for c in chosen if c in judged),
        "seed": seed,
        "worklist": str(out / "worklist.jsonl"),
        "audio": str(audio_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, help="Staged scenario bundle to sample from.")
    parser.add_argument("--out", type=Path, help="Directory to write the worklist and audio to.")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE, help="Clips to draw.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Sampling seed.")
    parser.add_argument(
        "--summary", action="store_true", help="Summarise existing verdicts and exit."
    )
    parser.add_argument(
        "--verdicts",
        type=Path,
        default=DEFAULT_VERDICTS_PATH,
        help=f"Verdict fixture (default: {DEFAULT_VERDICTS_PATH}).",
    )
    args = parser.parse_args()

    if args.summary:
        report = summarise_verdicts(read_verdicts(args.verdicts))
    elif args.bundle and args.out:
        report = _stage(args.bundle, args.out, args.size, args.seed)
    else:
        parser.error("--bundle and --out are required unless --summary is given")

    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

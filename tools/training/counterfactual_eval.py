"""Score the tashkeel counterfactual recordings: does the model hear, or reconstruct?

:mod:`training.minimal_pairs` showed that this corpus cannot answer the question. On the
held-out words whose consonant skeleton is genuinely ambiguous, a **text-only** baseline that
hears nothing scores 0.9734 — above every checkpoint — so the fine-tune's ~0.98 short-vowel
recall is fully explainable without the model hearing a single harakah. Every clip in the
corpus is correct recitation, so no observation in it separates the two hypotheses.

The counterfactual recordings do. Each item was recited twice by the same reciter: once as
written (``control``) and once with the target word's single short vowel replaced
(``counterfactual``). Where the spoken vowel and the canonical vowel disagree, the hypotheses
predict opposite things:

* a model that **hears** transcribes the vowel that was *spoken*,
* a model that **reconstructs** transcribes the vowel the canonical *text* prescribes.

The control take is what makes a negative result interpretable. If the model cannot render
the target word's vowel correctly even when it is recited correctly, that item measures the
model's general accuracy on this voice, not its response to the substitution — so it is
**dropped** rather than counted as a failure to follow the audio.

Alignment is run twice, against the canonical reference and against the counterfactual one
(the same string with the target vowel swapped). Aligning only against the canonical text
could bias the projection toward the canonical answer, which is precisely the hypothesis
under test; items where the two alignments disagree are reported separately rather than
being silently resolved in either direction.

Usage::

    python -m training.counterfactual_eval \
        --items tadabur/tashkeel_counterfactual_fixtures/counterfactual_items.jsonl \
        --manifest tadabur/audit_run/seg_v21/manifest_raw.jsonl \
        --audio-dir tadabur/audit_run/counterfactual_audio \
        --model tadabur/audit_run/seg_v21/rung3_v2/merged \
        --out tadabur/audit_run/seg_v21/counterfactual_rung3_v2.json
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np

from tadabur.smith_waterman import smith_waterman
from training.minimal_pairs import SHORT_VOWELS, read_segments

# What the model wrote for the target word, relative to the two competing predictions.
FOLLOWED_AUDIO = "followed_audio"
FOLLOWED_TEXT = "followed_text"
OTHER_VOWEL = "other_vowel"
NO_VOWEL = "no_vowel"


def wilson_interval(successes: int, total: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval — usable at the extreme proportions this test expects.

    The normal approximation degenerates near 0 and 1, which is exactly where a decisive
    result lands, so it would report a nonsensically narrow or out-of-range interval on the
    outcome that matters most.
    """
    if total == 0:
        return (0.0, 0.0)
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return (round(max(0.0, centre - margin), 12), round(min(1.0, centre + margin), 12))


def substitute_vowel(reference: str, start: int, end: int, vowel: str) -> str:
    """``reference`` with the single short vowel in ``[start, end)`` replaced by ``vowel``."""
    word = "".join(vowel if c in SHORT_VOWELS else c for c in reference[start:end])
    return reference[:start] + word + reference[end:]


def vowel_in_span(decode: str, reference: str, start: int, end: int) -> str | None:
    """The vowel the decode placed on the reference word at ``[start, end)``.

    Returns ``None`` when the alignment never reached the word — a word the model did not
    transcribe is not evidence about which vowel it heard.
    """
    alignment = smith_waterman(decode, reference)
    positions = [
        query
        for offset, query in enumerate(alignment.ref_to_query)
        if start <= alignment.ref_start + offset < end and query is not None and query >= 0
    ]
    if not positions:
        return None
    span = decode[min(positions) : max(positions) + 1]
    vowels = [c for c in span if c in SHORT_VOWELS]
    return vowels[0] if vowels else ""


@dataclass(frozen=True)
class TakeResult:
    """How one recorded take rendered the target word's vowel."""

    decoded_vowel: str | None
    decoded_vowel_alt: str | None

    @property
    def reached(self) -> bool:
        return self.decoded_vowel is not None

    @property
    def stable(self) -> bool:
        """Whether both alignment references agree on what the model wrote."""
        return self.decoded_vowel == self.decoded_vowel_alt


def classify(vowel: str | None, canonical: str, spoken: str) -> str:
    """Which hypothesis the rendered vowel supports."""
    if not vowel:
        return NO_VOWEL
    if vowel == spoken:
        return FOLLOWED_AUDIO
    if vowel == canonical:
        return FOLLOWED_TEXT
    return OTHER_VOWEL


def score_item(item: dict, segment: dict, decodes: dict[str, str]) -> dict:
    """One item's verdict, given its control and counterfactual decodes."""
    offsets = segment["raw_word_offsets"]
    index = item["word_index"]
    start, end = offsets[index], offsets[index + 1]
    canonical_reference = segment["raw_reference_phonemes"]
    spoken_reference = substitute_vowel(canonical_reference, start, end, item["spoken_vowel"])

    takes = {}
    for take, reference_first in (("control", canonical_reference), ("counterfactual", spoken_reference)):
        reference_second = spoken_reference if take == "control" else canonical_reference
        takes[take] = TakeResult(
            decoded_vowel=vowel_in_span(decodes[take], reference_first, start, end),
            decoded_vowel_alt=vowel_in_span(decodes[take], reference_second, start, end),
        )

    canonical, spoken = item["canonical_vowel"], item["spoken_vowel"]
    control, counterfactual = takes["control"], takes["counterfactual"]
    # The control take must show the model rendering this word's vowel correctly in this
    # voice; otherwise the counterfactual take measures general inaccuracy, not hearing.
    control_ok = control.decoded_vowel == canonical

    return {
        "item_id": item["item_id"],
        "surah_ayah": item["surah_ayah"],
        "target_word": item["target_word"],
        "canonical_vowel": canonical,
        "spoken_vowel": spoken,
        "swap": f"{canonical}->{spoken}",
        "control_vowel": control.decoded_vowel,
        "counterfactual_vowel": counterfactual.decoded_vowel,
        "control_passed": control_ok,
        "alignment_stable": control.stable and counterfactual.stable,
        "outcome": classify(counterfactual.decoded_vowel, canonical, spoken),
        "scored": control_ok,
    }


def summarize(results: list[dict]) -> dict:
    """Aggregate the scorable items into the headline answer."""
    scored = [r for r in results if r["scored"]]
    outcomes = {
        key: sum(1 for r in scored if r["outcome"] == key)
        for key in (FOLLOWED_AUDIO, FOLLOWED_TEXT, OTHER_VOWEL, NO_VOWEL)
    }
    total = len(scored)
    followed = outcomes[FOLLOWED_AUDIO]
    low, high = wilson_interval(followed, total)

    by_swap = {}
    for swap in sorted({r["swap"] for r in scored}):
        rows = [r for r in scored if r["swap"] == swap]
        by_swap[swap] = {
            "scored": len(rows),
            "followed_audio": sum(1 for r in rows if r["outcome"] == FOLLOWED_AUDIO),
            "followed_text": sum(1 for r in rows if r["outcome"] == FOLLOWED_TEXT),
        }

    return {
        "items": len(results),
        "control_failures_dropped": sum(1 for r in results if not r["control_passed"]),
        "alignment_unstable": sum(1 for r in results if not r["alignment_stable"]),
        "scored": total,
        "outcomes": outcomes,
        "followed_audio_rate": round(followed / total, 4) if total else None,
        "followed_audio_ci95": [round(low, 4), round(high, 4)],
        "by_swap": by_swap,
    }


def verdict(summary: dict, margin: float = 0.5) -> dict:
    """Does the model follow the audio or the canonical text?

    Judged on the confidence interval rather than the point estimate: with a few dozen items
    a bare proportion can look decisive when it is not. ``margin`` is the boundary between
    the two hypotheses — a hearing model should sit well above it, a reconstructing one well
    below — and the interval must clear it entirely for the result to count as settled.
    """
    if summary["scored"] < 20:
        return {
            "conclusive": False,
            "reason": f"only {summary['scored']} scorable items — too few to rule",
        }
    low, high = summary["followed_audio_ci95"]
    if low > margin:
        return {"conclusive": True, "hears_tashkeel": True,
                "interpretation": "the model transcribes the vowel that was spoken, not the "
                                  "one the canonical text prescribes — it is hearing tashkeel"}
    if high < margin:
        return {"conclusive": True, "hears_tashkeel": False,
                "interpretation": "the model transcribes the canonical vowel even when a "
                                  "different one was spoken — it is reconstructing from text, "
                                  "and cannot flag a student's vowel error"}
    return {
        "conclusive": False,
        "reason": f"95% CI [{low}, {high}] spans {margin} — more items needed to separate "
                  "hearing from reconstruction",
    }


def _decode_takes(model_id: str, items: list[dict], audio_dir: Path, batch_size: int,
                  device: str) -> dict[str, dict[str, str]]:
    """Decode both takes of every item, keyed ``item_id`` then take."""
    from tadabur.audio import TARGET_SAMPLE_RATE, decode_to_mono_16k
    from tadabur.inference import MuaalemPhonemeModel

    jobs = [(item["item_id"], take) for item in items for take in ("control", "counterfactual")]
    model = MuaalemPhonemeModel.load(model_id, device=device)
    decodes: dict[str, dict[str, str]] = {item["item_id"]: {} for item in items}
    for start in range(0, len(jobs), batch_size):
        chunk = jobs[start : start + batch_size]
        waves = [
            np.asarray(
                decode_to_mono_16k((audio_dir / f"{item_id}_{take}.wav").read_bytes()),
                dtype=np.float32,
            )
            for item_id, take in chunk
        ]
        for (item_id, take), result in zip(chunk, model.decode_batch(waves, TARGET_SAMPLE_RATE)):
            decodes[item_id][take] = result.phonemes
    del model
    return decodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--audio-dir", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    items = [json.loads(line) for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
    segments = {row["audio_filename"]: row for row in read_segments(args.manifest)}
    missing = [
        f"{item['item_id']}_{take}.wav"
        for item in items
        for take in ("control", "counterfactual")
        if not (args.audio_dir / f"{item['item_id']}_{take}.wav").is_file()
    ]
    if missing:
        raise SystemExit(f"{len(missing)} recordings missing, e.g. {missing[:3]}")

    decodes = _decode_takes(args.model, items, args.audio_dir, args.batch_size, args.device)
    results = [score_item(item, segments[item["audio_filename"]], decodes[item["item_id"]])
               for item in items]

    report = {
        "model": args.model,
        "summary": summarize(results),
        "items": results,
    }
    report["verdict"] = verdict(report["summary"], args.margin)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "verdict": report["verdict"]},
                     indent=2, ensure_ascii=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

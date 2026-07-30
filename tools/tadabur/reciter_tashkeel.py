"""Reciter-layer tashkeel filter (ADR-0005).

The ADR-0001 gate is computed on ``normalize_phonemes``, which deletes short vowels, so it
cannot have rejected a single clip for wrong tashkeel — a recitation with *every* vowel wrong
scores a perfect ``match_ratio``. Nothing else in the pipeline looks at vowels either, so
whatever tashkeel error the corpus contains reached the training labels untouched.

ADR-0003 rejected a per-vowel colour-swap gate at the **segment** level and that rejection
stands: one segment carries too few vowels to separate a reciter's error from the same model's
own noise, and dropping those segments would preferentially delete the omission-correction
examples the fine-tune needs. A **reciter** is different — thousands of vowels aggregate, so a
systematically non-Hafs reciter becomes separable where a single segment never was.

Two design choices keep this honest:

* **Swap, not omission.** ``swapped`` means the model confidently decoded a *different* vowel
  than Hafs prescribes **on a carrier consonant it also got right**, which is what a different
  qira'ah produces. A wrong vowel on a *misheard* consonant is a decode failure and is counted
  separately (``unanchored_wrong``), or bad audio would look like bad recitation. ``omitted``
  means the model declined to commit; it varies 3.6x across reciters and tracks audio quality,
  pace and accent, so gating on it would delete hard-but-correct audio.
* **The Wilson lower bound, not the point estimate.** A reciter with 40 vowels and one unlucky
  swap sits at 2.5%, far above any sane threshold. Requiring the *lower* bound to clear it means
  a reciter is only excluded when the evidence supports it.

On the corpus as it stands this excludes nobody — swap rate is median 0.0000 / max 0.0056 over
the 163 reciters clearing the evidence floor, who between them carry ~91% of the corpus's
reference vowels, while the default threshold is 1%. That is the finding, not a failure: no
high-swap population exists among the reciters supplying almost all the data. It is *not* proof
that the corpus is free of qira'ah contamination — 339 low-volume reciters go unjudged, and the
threshold has no validated positive control. The module ships as a standing guard so the next
corpus does not have to rediscover it.

Torch-free: it reads a scored manifest and writes a report, no model and no GPU.

    python -m tadabur.reciter_tashkeel \\
        --manifest audit_run/seg_v21/manifest_raw.jsonl \\
        --out audit_run/seg_v21/reciter_tashkeel.json [--max-swap-rate 0.01]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

from training.tashkeel_eval import VowelCounts, score_vowels

# A reciter must clear this many reference vowels before the filter will judge them at all.
# Below it the confidence bound is so wide that no threshold can be cleared anyway, and saying
# "insufficient evidence" is more useful than an exclusion that reflects sample size.
MIN_REFERENCE_VOWELS = 500

# Ceiling on a reciter's confidently-wrong-vowel rate. The corpus baseline is 0.0005 and the
# worst observed reciter is 0.0056, so 1% sits an order of magnitude clear of reciter noise.
# It has NOT been validated against a known non-Hafs positive control, so it is calibrated to
# separate from observed noise, not proven to catch a real qira'ah difference. Treat a
# non-zero exclusion as a prompt to investigate, never as a proven verdict.
MAX_SWAP_RATE = 0.01


def wilson_lower_bound(successes: int, total: int, z: float = 1.96) -> float:
    """Lower 95% bound on a proportion.

    The *lower* bound is deliberate: a reciter is excluded only when the evidence supports the
    claim, so a small sample can never trigger an exclusion it has not earned.
    """
    if total == 0:
        return 0.0
    p = successes / total
    denominator = 1 + z * z / total
    centre = (p + z * z / (2 * total)) / denominator
    margin = z * sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return max(0.0, centre - margin)


@dataclass(frozen=True)
class ReciterVerdict:
    """One reciter's tashkeel behaviour and whether it disqualifies them."""

    reciter_id: str
    counts: VowelCounts
    swap_rate: float
    swap_rate_lower95: float
    judged: bool
    excluded: bool

    def to_dict(self) -> dict:
        return {
            "reciter_id": self.reciter_id,
            "reference_vowels": self.counts.reference_total,
            "swapped": self.counts.swapped,
            "swap_rate": round(self.swap_rate, 6),
            "swap_rate_lower95": round(self.swap_rate_lower95, 6),
            "judged": self.judged,
            "excluded": self.excluded,
        }


def score_reciters(rows) -> dict[str, VowelCounts]:
    """Aggregate every row's vowel outcomes per reciter.

    Rows carry the model decode and the tashkeel-bearing reference; the *raw* reference is
    required, since the normalized one has had the vowels stripped out of it.
    """
    per: dict[str, VowelCounts] = defaultdict(VowelCounts)
    for row in rows:
        per[str(row["reciter_id"])] += score_vowels(
            row["predicted_phonemes"], row["raw_reference_phonemes"]
        )
    return dict(per)


def judge(
    per_reciter: dict[str, VowelCounts],
    max_swap_rate: float = MAX_SWAP_RATE,
    min_vowels: int = MIN_REFERENCE_VOWELS,
) -> list[ReciterVerdict]:
    """Rule on each reciter, worst first."""
    verdicts = []
    for reciter_id, counts in per_reciter.items():
        total = counts.reference_total
        rate = counts.swapped / total if total else 0.0
        lower = wilson_lower_bound(counts.swapped, total)
        judged = total >= min_vowels
        verdicts.append(
            ReciterVerdict(
                reciter_id=reciter_id,
                counts=counts,
                swap_rate=rate,
                swap_rate_lower95=lower,
                judged=judged,
                excluded=judged and lower > max_swap_rate,
            )
        )
    return sorted(verdicts, key=lambda v: (-v.swap_rate, v.reciter_id))


def summarize(verdicts: list[ReciterVerdict], max_swap_rate: float = MAX_SWAP_RATE) -> dict:
    """The corpus-level picture, including whether an outlier population exists at all."""
    judged = [v for v in verdicts if v.judged]
    rates = sorted(v.swap_rate for v in judged)
    pooled = VowelCounts()
    for v in verdicts:
        pooled += v.counts

    def percentile(fraction: float) -> float | None:
        if not rates:
            return None
        return rates[min(len(rates) - 1, int(fraction * len(rates)))]

    return {
        "reciters": len(verdicts),
        "judged": len(judged),
        "excluded": sum(1 for v in judged if v.excluded),
        "max_swap_rate": max_swap_rate,
        "min_reference_vowels": MIN_REFERENCE_VOWELS,
        "corpus_swap_rate": (
            round(pooled.swapped / pooled.reference_total, 6) if pooled.reference_total else None
        ),
        "swap_rate_median": percentile(0.5),
        "swap_rate_p90": percentile(0.9),
        "swap_rate_max": rates[-1] if rates else None,
        "excluded_reciters": [v.reciter_id for v in judged if v.excluded],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-swap-rate", type=float, default=MAX_SWAP_RATE)
    parser.add_argument("--min-vowels", type=int, default=MIN_REFERENCE_VOWELS)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    rows = (
        json.loads(line)
        for line in args.manifest.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    verdicts = judge(score_reciters(rows), args.max_swap_rate, args.min_vowels)
    report = {
        "summary": summarize(verdicts, args.max_swap_rate),
        "reciters": [v.to_dict() for v in verdicts],
    }
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, ensure_ascii=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

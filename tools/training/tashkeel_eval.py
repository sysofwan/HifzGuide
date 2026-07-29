"""Tashkeel-sensitive phoneme eval — the gate ADR-0003 needs and the repo never had (#10).

ADR-0003's whole purpose is to raise the model's **short-vowel** (fatha/damma/kasra)
reliability, and the fine-tune trains on the tashkeel-bearing ``raw_reference_phonemes`` to
that end. Yet **no other gate in this repo can see a short vowel.**
:func:`tadabur.normalization.normalize_phonemes` strips ``U+064E``/``U+064F``/``U+0650``
unconditionally — there is only one normalization and the ``.strict`` scorer uses it too — so
a decode with full tashkeel and the same decode with every vowel deleted score *identically*
against the same reference. That blindness is not hypothetical: a fine-tune that emitted
**zero** short vowels (the ADR-0003 label defect) passed every #10 gate.

This module is the missing measurement. It runs the **real checkpoint on real audio** and
compares the decode against the raw reference **without normalizing either side**, so a
vowel is a first-class outcome:

* **matched** — the reference vowel is aligned to the same vowel in the decode.
* **swapped** — aligned to a *different* short vowel (confident-wrong i'raab, the dangerous
  error: it asserts the wrong case ending rather than declining to guess).
* **omitted** — aligned to a gap or a non-vowel (the model declined to mark the vowel).
* **spurious** — a decode vowel with no reference vowel behind it.

Alignment is :func:`tadabur.smith_waterman.smith_waterman` over the *un-normalized* strings,
using its full column sequence so decode-side insertions stay visible (the reference-indexed
views hide them, which would make ``spurious`` unobservable).

**The gate is a no-regression gate, not an absolute-quality one.** The base Muaalem checkpoint
already emits tashkeel; the risk this must catch is a fine-tune *destroying* that capability,
which is exactly what happened. So a candidate is judged against the **base model scored on
the same windows** (:data:`DEFAULT_REGRESSION_TOLERANCE`), and an absolute floor guards the
degenerate case where both are bad.

Windows come from the **validation split** of the windowed labels — reciter-disjoint from
train by construction (:func:`training.windowed_labels.split_by_reciter`), and carrying the
exact ``[start_sample, num_samples)`` training geometry, so the gate measures the model on the
unit it was actually trained on.

Runs on Linux + CUDA (see ``tools/README.md``).

Usage:
  python -m training.tashkeel_eval \\
      --labels audit_run/seg_v21/windowed_labels_v2.jsonl \\
      --audio-dir audit_run/clips_v2 \\
      --model audit_run/seg_v21/rung3/merged \\
      --out audit_run/seg_v21/tashkeel_eval.json
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from tadabur.smith_waterman import smith_waterman

#: The Muaalem phoneme head's three short-vowel output classes (ids 32/33/34).
FATHA, DAMMA, KASRA = "\u064e", "\u064f", "\u0650"
SHORT_VOWELS = frozenset({FATHA, DAMMA, KASRA})

#: Absolute floor on vowel recall. Deliberately low: this catches collapse (a head that has
#: stopped emitting vowels at all), not marginal quality, which the regression check owns.
DEFAULT_MIN_RECALL = 0.50

#: How far below the base model's recall a candidate may fall before it counts as a
#: regression. The fine-tune is allowed to trade a little tashkeel for waqf/phoneme gains,
#: but not to lose the capability.
DEFAULT_REGRESSION_TOLERANCE = 0.05


@dataclass(frozen=True)
class VowelCounts:
    """Reference-vowel outcomes plus decode-side spurious vowels, over some window set."""

    matched: int = 0
    swapped: int = 0
    omitted: int = 0
    spurious: int = 0
    unanchored: int = 0

    def __add__(self, other: "VowelCounts") -> "VowelCounts":
        return VowelCounts(
            self.matched + other.matched,
            self.swapped + other.swapped,
            self.omitted + other.omitted,
            self.spurious + other.spurious,
            self.unanchored + other.unanchored,
        )

    @property
    def reference_total(self) -> int:
        return self.matched + self.swapped + self.omitted + self.unanchored

    @property
    def recall(self) -> float:
        """Share of reference vowels the decode reproduced with the right colour."""
        return self.matched / self.reference_total if self.reference_total else 0.0

    @property
    def precision(self) -> float:
        """Share of decoded vowels that were correct."""
        emitted = self.matched + self.swapped + self.spurious
        return self.matched / emitted if emitted else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if p + r else 0.0

    @property
    def swap_rate(self) -> float:
        """Share of reference vowels given a *confidently wrong* colour.

        Tracked separately from omission because the two are not equally harmful: an omitted
        vowel leaves the scorer no signal, while a swapped one asserts the wrong i'raab.
        """
        return self.swapped / self.reference_total if self.reference_total else 0.0


def score_vowels(decode: str, reference: str) -> VowelCounts:
    """Classify every short vowel in ``reference`` (and every spurious one in ``decode``).

    Both strings are the **raw, un-normalized** phoneme forms — normalizing either side
    would delete the very characters being measured. Uses the aligner's complete column
    sequence so a decode-only column (an insertion) is visible as ``spurious``.

    A vowel only counts as ``matched`` when its **carrier consonant also matched**. Without
    that anchor the metric is trivially gameable: Smith-Waterman is local and will happily
    gap over every consonant, so a "decode" consisting of nothing but the reference's vowel
    sequence scored a perfect 1.000 recall *and* 1.000 precision. Requiring the carrier
    makes the metric mean "the right vowel on the right consonant", which is the capability
    ADR-0003 actually cares about. Correct-colour vowels on an unheard carrier are counted
    as ``unanchored`` rather than credited.
    """
    if not reference:
        return VowelCounts()
    alignment = smith_waterman(decode, reference)

    counts = VowelCounts()
    aligned_reference_vowels = 0
    carrier_matched = False
    for column in alignment.columns:
        ref_char, query_char = column.ref_char, column.query_char
        if ref_char is not None and ref_char not in SHORT_VOWELS:
            # The consonant (or long vowel) this harakah will sit on.
            carrier_matched = query_char == ref_char
        if ref_char in SHORT_VOWELS:
            aligned_reference_vowels += 1
            if query_char == ref_char:
                counts += VowelCounts(matched=1) if carrier_matched else VowelCounts(unanchored=1)
            elif query_char in SHORT_VOWELS:
                counts += VowelCounts(swapped=1)
            else:
                counts += VowelCounts(omitted=1)
        elif query_char in SHORT_VOWELS:
            # ref_char is None (an insertion) or a non-vowel the model voweled anyway.
            # Both are vowels emitted with no reference vowel behind them.
            counts += VowelCounts(spurious=1)

    # Smith-Waterman is *local*, so both strings have unaligned ends that produced no column
    # at all. Each side needs charging, or the metric flatters the model twice over.
    #
    # Reference side: those vowels were expected and not delivered, so counting only the
    # aligned span would let a decode matching a short fragment score a perfect recall.
    unaligned = sum(1 for c in reference if c in SHORT_VOWELS) - aligned_reference_vowels
    # Decode side: symmetrically, a vowel the model invented outside the aligned span is
    # still an emitted vowel with no reference behind it. Ignoring it inflates precision --
    # the trimmed ends are exactly where a hallucinated vowel is most likely to appear.
    edge_spurious = sum(
        1 for c in decode[:alignment.query_start] + decode[alignment.query_end:]
        if c in SHORT_VOWELS
    )
    return counts + VowelCounts(omitted=max(0, unaligned), spurious=edge_spurious)


@dataclass(frozen=True)
class TashkeelReport:
    """One checkpoint's vowel behaviour over the scored windows."""

    model: str
    windows: int
    counts: VowelCounts
    per_vowel: dict[str, VowelCounts]

    def to_dict(self) -> dict:
        return {
            "model": self.model,
            "windows": self.windows,
            "recall": round(self.counts.recall, 4),
            "precision": round(self.counts.precision, 4),
            "f1": round(self.counts.f1, 4),
            "swap_rate": round(self.counts.swap_rate, 4),
            "counts": asdict(self.counts),
            "per_vowel": {
                name: {
                    "recall": round(c.recall, 4),
                    "swap_rate": round(c.swap_rate, 4),
                    **asdict(c),
                }
                for name, c in self.per_vowel.items()
            },
        }


def score_windows(decodes: list[str], references: list[str], model: str) -> TashkeelReport:
    """Aggregate :func:`score_vowels` over paired decode/reference windows.

    ``per_vowel`` re-scores restricted to one vowel at a time so a model that reproduces
    fatha well but never marks kasra cannot hide behind the pooled number.
    """
    if len(decodes) != len(references):
        raise ValueError(f"{len(decodes)} decodes vs {len(references)} references")

    total = VowelCounts()
    for decode, reference in zip(decodes, references):
        total += score_vowels(decode, reference)

    per_vowel = {
        name: _score_single_vowel(decodes, references, vowel)
        for name, vowel in (("fatha", FATHA), ("damma", DAMMA), ("kasra", KASRA))
    }
    return TashkeelReport(model, len(decodes), total, per_vowel)


def _score_single_vowel(decodes: list[str], references: list[str], vowel: str) -> VowelCounts:
    """:func:`score_vowels` restricted to reference positions holding ``vowel``.

    Alignment still runs on the full strings — dropping the other vowels first would change
    the alignment and measure a different model.
    """
    total = VowelCounts()
    for decode, reference in zip(decodes, references):
        if vowel not in reference:
            continue
        alignment = smith_waterman(decode, reference)
        aligned = 0
        counts = VowelCounts()
        for column in alignment.columns:
            if column.ref_char != vowel:
                continue
            aligned += 1
            if column.query_char == vowel:
                counts += VowelCounts(matched=1)
            elif column.query_char in SHORT_VOWELS:
                counts += VowelCounts(swapped=1)
            else:
                counts += VowelCounts(omitted=1)
        unaligned = reference.count(vowel) - aligned
        total += counts + VowelCounts(omitted=max(0, unaligned))
    return total


def gate(
    candidate: TashkeelReport,
    baseline: TashkeelReport | None,
    min_recall: float = DEFAULT_MIN_RECALL,
    tolerance: float = DEFAULT_REGRESSION_TOLERANCE,
) -> dict:
    """Pass/fail verdict: an absolute floor plus no-regression against ``baseline``.

    ``baseline`` is the *base* checkpoint scored on the same windows. Without it only the
    floor can be applied, and the verdict records that the regression check was skipped —
    the check that actually catches a destroyed capability, so a missing baseline is a
    weaker verdict, never a silent pass.

    Pooled recall alone is not sufficient. ADR-0003 requires this eval to catch "aggregate
    vowel accuracy improving while that discrimination collapses", so three further
    regressions fail the gate independently of the pooled number:

    * **swap rate rising** — a confidently *wrong* i'raab is the poisonous failure; a model
      may trade omissions for swaps and leave recall flattered.
    * **precision falling** — recall is trivially maxed by voweling everything.
    * **a single colour collapsing** — kasra is the weakest class, so it can be sacrificed
      while fatha's larger count holds the pooled recall up.
    """
    meets_floor = candidate.counts.recall >= min_recall
    regressed_recall = (
        baseline is not None
        and candidate.counts.recall < baseline.counts.recall - tolerance
    )
    regressed_swap = (
        baseline is not None
        and candidate.counts.swap_rate > baseline.counts.swap_rate + tolerance
    )
    regressed_precision = (
        baseline is not None
        and candidate.counts.precision < baseline.counts.precision - tolerance
    )
    collapsed = sorted(
        name for name, base in (baseline.per_vowel.items() if baseline else ())
        if base.reference_total
        and candidate.per_vowel.get(name, VowelCounts()).recall < base.recall - tolerance
    )
    regressed = bool(
        regressed_recall or regressed_swap or regressed_precision or collapsed
    )
    return {
        "passed": bool(meets_floor and not regressed),
        "meets_floor": bool(meets_floor),
        "min_recall": min_recall,
        "regressed_vs_baseline": regressed,
        "regressed_recall": bool(regressed_recall),
        "regressed_swap_rate": bool(regressed_swap),
        "regressed_precision": bool(regressed_precision),
        "collapsed_vowels": collapsed,
        "regression_tolerance": tolerance,
        "baseline_compared": baseline is not None,
        "candidate_recall": round(candidate.counts.recall, 4),
        "baseline_recall": round(baseline.counts.recall, 4) if baseline else None,
        "candidate_swap_rate": round(candidate.counts.swap_rate, 4),
        "baseline_swap_rate": round(baseline.counts.swap_rate, 4) if baseline else None,
        "candidate_precision": round(candidate.counts.precision, 4),
        "baseline_precision": round(baseline.counts.precision, 4) if baseline else None,
        "recall_delta": (
            round(candidate.counts.recall - baseline.counts.recall, 4) if baseline else None
        ),
    }


def _load_windows(labels_path: Path, split: str, limit: int | None):
    """The split's window labels (clip, span, raw phoneme label), in file order."""
    from training.windowed_labels import read_labels

    by_split = read_labels(labels_path)
    labels = by_split.get(split, [])
    if not labels:
        raise ValueError(
            f"{labels_path} has no '{split}' windows (splits: {sorted(by_split)})."
        )
    if not limit or limit >= len(labels):
        return labels
    # Labels are written in filename order, so labels[:limit] is a *contiguous* slice --
    # 400 of 5,227 val windows turned out to cover only 2 of 45 reciters, and the report
    # said nothing about it. Sample deterministically across the whole split instead.
    sampled = random.Random(0).sample(labels, limit)
    return sorted(sampled, key=lambda w: (w.clip_audio_filename, w.window_index))


def coverage_of(labels) -> dict:
    """What the scored sample actually spans, so a partial eval can never look total."""
    return {
        "windows": len(labels),
        "reciters": len({w.reciter_id for w in labels}),
        "clips": len({w.clip_audio_filename for w in labels}),
        "ayahs": len({w.surah_ayah for w in labels}),
    }


def _decode_windows(model_id: str, labels, audio_dir: Path, batch_size: int, device: str):
    """Decode each window's exact training audio span with ``model_id``."""
    from tadabur.audio import TARGET_SAMPLE_RATE
    from tadabur.inference import MuaalemPhonemeModel
    from training.windowed_batch import ClipAudioCache

    cache = ClipAudioCache(audio_dir)
    model = MuaalemPhonemeModel.load(model_id, device=device)
    decodes: list[str] = []
    for start in range(0, len(labels), batch_size):
        chunk = labels[start : start + batch_size]
        waves = []
        for label in chunk:
            waveform = cache.waveform(label.clip_audio_filename)
            end = label.start_sample + label.num_samples
            waves.append(np.asarray(waveform[label.start_sample : end], dtype=np.float32))
        decodes.extend(d.phonemes for d in model.decode_batch(waves, TARGET_SAMPLE_RATE))
    del model
    return decodes


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True,
                        help="windowed CTC labels JSONL (training.windowed_labels).")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="staged 16 kHz clip directory.")
    parser.add_argument("--model", required=True,
                        help="candidate checkpoint (merged model dir or hub id).")
    parser.add_argument("--baseline", default="obadx/muaalem-model-v3_2",
                        help="baseline checkpoint scored on the same windows; "
                             "'none' to skip the no-regression check.")
    parser.add_argument("--split", default="val",
                        help="label split to score (default: the held-out val split).")
    parser.add_argument("--limit", type=int, default=0,
                        help="score at most this many windows, sampled deterministically "
                             "across the whole split (default 0 = all of it).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--min-recall", type=float, default=DEFAULT_MIN_RECALL)
    parser.add_argument("--tolerance", type=float, default=DEFAULT_REGRESSION_TOLERANCE)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    labels = _load_windows(args.labels, args.split, args.limit or None)
    references = [label.phoneme_label for label in labels]
    if not any(c in SHORT_VOWELS for ref in references for c in ref):
        raise ValueError(
            f"{args.labels} '{args.split}' labels contain no short vowels — they were built "
            "from the normalized reference (ADR-0003). Rebuild from raw_reference_phonemes; "
            "scoring tashkeel against a vowel-free reference would report a vacuous pass."
        )
    coverage = coverage_of(labels)
    print(
        f"Scoring {coverage['windows']} '{args.split}' windows — "
        f"{coverage['reciters']} reciters, {coverage['clips']} clips, "
        f"{coverage['ayahs']} ayahs.",
        flush=True,
    )

    candidate = score_windows(
        _decode_windows(args.model, labels, args.audio_dir, args.batch_size, args.device),
        references, args.model,
    )
    baseline = None
    if args.baseline.lower() != "none":
        baseline = score_windows(
            _decode_windows(
                args.baseline, labels, args.audio_dir, args.batch_size, args.device
            ),
            references, args.baseline,
        )

    verdict = gate(candidate, baseline, args.min_recall, args.tolerance)
    report = {
        "coverage": coverage,
        "candidate": candidate.to_dict(),
        "baseline": baseline.to_dict() if baseline else None,
        "gate": verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(json.dumps(verdict, indent=2))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

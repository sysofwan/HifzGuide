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
from math import comb, sqrt
from pathlib import Path

import numpy as np

from tadabur.smith_waterman import smith_waterman
from training.minimal_pairs import SHORT_VOWELS, read_segments

# What the model wrote for the target word, relative to the two competing predictions.
FOLLOWED_AUDIO = "followed_audio"
FOLLOWED_TEXT = "followed_text"
OTHER_VOWEL = "other_vowel"
NO_VOWEL = "no_vowel"

# Fraction of the target word's consonant skeleton the alignment must match before the
# projected span is trusted. Without this, a repeated word elsewhere in the ayah can capture
# the span on a couple of incidental characters and return the wrong word's vowel.
MIN_SKELETON_COVERAGE = 0.5

# Ceiling on the silent-correction rate: how often a genuinely wrong vowel is transcribed as
# the canonical one, an error the student is never told about. Reported as context, NOT as the
# gate — Muraja deliberately relaxes the base model's strictness, so an absolute bar would
# fail the fine-tune for succeeding at its actual goal. See ADR-0003.
MAX_SILENT_CORRECTION_RATE = 0.05

# The real gate (ADR-0003): how much tashkeel discrimination the fine-tune may lose relative
# to base. Relaxing madd and similar over-strict phenomena is the point; letting that
# relaxation bleed into tashkeel is the failure. Taken against the upper confidence bound.
#
# This margin, NOT the choice of interval, is the operative lever (ADR-0006). At the sample
# sizes this eval can reach, one discordant item is worth ~2.4%, so the margin is barely
# coarser than the measurement's own resolution. It is a product tolerance -- how often a
# student's wrong vowel may go unflagged relative to base -- and must be argued as one.
MAX_REGRESSION = 0.05


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


def paired_score_interval(b: int, c: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Tango score interval for the paired net difference ``(b - c) / n``.

    ``b`` and ``c`` are the two discordant counts of a paired binary comparison and ``n`` the
    number of paired items; concordant items enter only through ``n``.

    A **Wald** interval must not be used here. Its width is proportional to ``b + c``, so at
    ``b == c == 0`` it collapses to zero and would certify non-inferiority on ten items — a
    looser rule than the one this replaces, which is the opposite of the intent. The score
    interval degrades gracefully instead: with no discordant pairs at all it reduces exactly
    to the Wilson bound ``z^2 / (n + z^2)``, so certification still costs sample size.

    When ``c == 0`` — the case this corpus actually presents, because the base model silently
    corrects nothing — the upper bound coincides with the Wilson bound on ``b / n``. That
    coincidence is why adopting this interval changes no verdict on the recorded sets; see
    ADR-0006. It matters only once recoveries exist to offset regressions.
    """
    if n <= 0:
        return (0.0, 0.0)

    def score(delta: float) -> float:
        numerator = b - c - n * delta
        a = 2 * n
        beta = -b - c + (2 * n - b + c) * delta
        gamma = -c * delta * (1 - delta)
        p21 = (-beta + sqrt(max(beta * beta - 4 * a * gamma, 0.0))) / (2 * a)
        variance = n * (2 * p21 + delta * (1 - delta))
        if variance <= 0:
            # Only reachable at the degenerate point where the interval is a single mass;
            # a zero numerator there is agreement with ``delta``, not infinite evidence.
            return 0.0 if numerator == 0 else (1 if numerator > 0 else -1) * float("inf")
        return numerator / sqrt(variance)

    def solve(accept) -> float:
        low, high = -1.0 + 1e-12, 1.0 - 1e-12
        for _ in range(60):
            mid = (low + high) / 2
            if accept(score(mid)):
                low = mid
            else:
                high = mid
        return low

    # ``score`` decreases in delta, so the interval is the band where it stays within +-z.
    upper = solve(lambda s: s >= -z)
    lower = solve(lambda s: s > z)
    return (round(max(-1.0, lower), 12), round(min(1.0, upper), 12))


def required_items(
    regression_rate: float,
    recovery_rate: float = 0.0,
    max_regression: float = MAX_REGRESSION,
    limit: int = 500_000,
) -> int | None:
    """Paired items needed to certify non-inferiority at an assumed discordance rate.

    Returns ``None`` when no sample size suffices. That is not a corner case: certification
    requires the *point* estimate ``regression_rate - recovery_rate`` to sit strictly below
    the margin, since the interval only ever shrinks onto it. A checkpoint whose observed net
    regression already exceeds the margin cannot be rescued by recording more audio, and a
    checkpoint sitting just under it needs an unreachable amount — which is the whole reason
    #60 withdrew the earlier "collect ~35 more items" advice.

    Rounding the rates to whole items makes certification non-monotone in ``n``: at a 2% rate
    and a 5% margin, 173 and 174 items clear the bound but 175 does not, because the third
    regression rounds up to a fourth. Reporting such an island as the answer would send a
    recollection to a sample size it could fall straight back out of, so this returns the
    start of the first **contiguous** run that certifies (202, there) instead.
    """
    if regression_rate - recovery_rate >= max_regression:
        return None

    def certifies(n: int) -> bool:
        b, c = round(regression_rate * n), round(recovery_rate * n)
        return paired_score_interval(b, c, n)[1] <= max_regression

    # The bound tightens with n, so double until it clears, then bisect. Rounding the two
    # rates to whole items makes that only near-monotone and leaves isolated islands that
    # certify while n+1 does not, so step back to the start of the run rather than reporting
    # the island. See the docstring for the 2%/173 case.
    n = 1
    while n <= limit and not certifies(n):
        n *= 2
    if n > limit:
        return None
    low, high = n // 2, n
    while low + 1 < high:
        mid = (low + high) // 2
        if certifies(mid):
            high = mid
        else:
            low = mid
    while high > 1 and certifies(high - 1):
        high -= 1
    return high


def substitute_vowel(reference: str, start: int, end: int, vowel: str) -> str:
    """``reference`` with the single short vowel in ``[start, end)`` replaced by ``vowel``."""
    word = "".join(vowel if c in SHORT_VOWELS else c for c in reference[start:end])
    return reference[:start] + word + reference[end:]


def vowel_in_span(decode: str, reference: str, start: int, end: int) -> str | None:
    """The vowel the decode placed on the reference word at ``[start, end)``.

    Returns ``None`` when the projection is not trustworthy, rather than guessing. Three
    things make it untrustworthy, and all three occur in practice:

    * the alignment never reached the word — a word the model did not transcribe is not
      evidence about which vowel it heard;
    * too little of the word's consonant skeleton was matched, which is how a repeated word
      elsewhere in the ayah can capture the span and hand back the wrong word's vowel;
    * the projected span carries more than one distinct short vowel, so which one belongs to
      the target is ambiguous.
    """
    alignment = smith_waterman(decode, reference)
    positions = [
        query
        for offset, query in enumerate(alignment.ref_to_query)
        if start <= alignment.ref_start + offset < end and query is not None and query >= 0
    ]
    if not positions:
        return None

    skeleton = [c for c in reference[start:end] if c not in SHORT_VOWELS and not c.isspace()]
    if skeleton and len(positions) < MIN_SKELETON_COVERAGE * len(skeleton):
        return None

    span = decode[min(positions) : max(positions) + 1]
    vowels = {c for c in span if c in SHORT_VOWELS}
    if len(vowels) > 1:
        return None
    return next(iter(vowels)) if vowels else ""


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


MADD_LETTERS = frozenset("\u0627\u0648\u064a\u06e5\u06e6\u0649")


def _is_madd_word(word: str) -> bool:
    """Whether a short vowel in ``word`` is held long by a following carrier.

    Such an item is not a valid probe: the reciter cannot say مَا as مُا, so the
    "counterfactual" take does not contain the vowel the sheet asked for.

    A **word-final** haraka is not madd and must not be excluded — ٱبْنُ, ٱدْعُ and
    ٱسْمَ carry an ordinary final vowel a reciter can freely change, and they are
    exactly the probes this harness needs. An earlier version treated the end of the
    word as if it were a carrier, silently discarding three valid items and shrinking
    the scorable set from 42 to 39.
    """
    return any(
        c in SHORT_VOWELS and i + 1 < len(word) and word[i + 1] in MADD_LETTERS
        for i, c in enumerate(word)
    )


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
    # Five items in the recorded set carry an elongation the reciter could not actually
    # substitute (فِى، ذُو، ذِى، ذَا، مَا). The generator now rejects these, but the audio
    # already exists, so they are dropped here rather than silently scored.
    excluded_madd = _is_madd_word(item["target_word"])

    stable = control.stable and counterfactual.stable

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
        "excluded_madd": excluded_madd,
        "alignment_stable": stable,
        "outcome": classify(counterfactual.decoded_vowel, canonical, spoken),
        # An unstable item is one the two alignment references disagree about, so its verdict
        # is an artifact of which reference was chosen -- and the two references here are the
        # canonical and the spoken text, exactly the hypotheses under test. Counting it would
        # let the answer depend on the question. It is reported, not scored.
        "scored": control_ok and not excluded_madd and stable,
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
    silent_low, silent_high = wilson_interval(outcomes[FOLLOWED_TEXT], total)

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
        "excluded_madd": sum(1 for r in results if r.get("excluded_madd")),
        "control_failures_dropped": sum(
            1 for r in results if not r["control_passed"] and not r.get("excluded_madd")
        ),
        "alignment_unstable": sum(1 for r in results if not r["alignment_stable"]),
        "scored": total,
        "outcomes": outcomes,
        "silent_correction_rate": round(outcomes[FOLLOWED_TEXT] / total, 4) if total else None,
        "silent_correction_ci95": [round(silent_low, 4), round(silent_high, 4)],
        "followed_audio_rate": round(followed / total, 4) if total else None,
        "followed_audio_ci95": [round(low, 4), round(high, 4)],
        "by_swap": by_swap,
    }


def compare_to_baseline(
    results: list[dict], baseline: list[dict], max_regression: float = MAX_REGRESSION
) -> dict:
    """Did the fine-tune lose vowel errors the base model still flagged?

    This is the gate ADR-0003 actually asks for. Muraja deliberately *relaxes* the base
    model's strictness — it ignores madd markers and other phenomena Muaalem over-flags — so
    an absolute miss-rate ceiling would fail the fine-tune for doing its job. What must not
    happen is the relaxation bleeding into tashkeel: "aggregate vowel accuracy improving
    while that discrimination collapses is the failure this eval must catch."

    Paired on item id, because both models saw identical audio; only the discordant items
    carry information. Reported as an exact McNemar test alongside the regression count, so a
    handful of discordant pairs cannot masquerade as a verdict.

    Non-inferiority is certified on the upper bound of the paired net difference
    ``(b - c) / n`` (ADR-0006), never on the point estimate and never on ``b`` alone.
    """
    mine = {r["item_id"]: r for r in results if r["scored"]}
    theirs = {r["item_id"]: r for r in baseline if r["scored"]}
    shared = sorted(set(mine) & set(theirs))

    def flags(row):
        return row["outcome"] != FOLLOWED_TEXT

    regressed = [i for i in shared if flags(theirs[i]) and not flags(mine[i])]
    recovered = [i for i in shared if not flags(theirs[i]) and flags(mine[i])]

    b, c = len(regressed), len(recovered)
    discordant = b + c
    p_value = (
        min(1.0, 2 * sum(comb(discordant, k) for k in range(min(b, c) + 1)) / 2**discordant)
        if discordant
        else 1.0
    )
    rate = b / len(shared) if shared else None
    _, high = wilson_interval(b, len(shared)) if shared else (None, None)
    needed = (
        required_items(b / len(shared), c / len(shared), max_regression) if shared else None
    )

    # Three-state, because a paired set this size can rarely prove non-inferiority. Requiring
    # BOTH directional evidence and magnitude keeps a single discordant item from reading as a
    # regression, while refusing to call a directional-but-underpowered result "clean".
    if b <= c:
        finding, detail = "no_evidence_of_regression", (
            f"the fine-tune silently corrected no more vowel errors than base ({b} vs {c})"
        )
    elif p_value < 0.05 and high > max_regression:
        finding, detail = "regression", (
            f"the fine-tune silently corrected {b} vowel errors base still flagged "
            f"(recovered {c}, exact p={p_value}) — the relaxation has reached tashkeel"
        )
    elif needed is None:
        # Underpowered but not fixable: the observed net regression is at or above the margin,
        # so the interval can never shrink under it. Calling this "more items needed" is what
        # #60 withdrew — it sends the project recording audio that cannot change the answer.
        finding, detail = "disqualified", (
            f"{b} regressed vs {c} recovered (exact p={p_value}) — the net regression "
            f"{(b - c) / len(shared):.1%} is not below the {max_regression:.0%} margin, so no "
            "sample size can certify this checkpoint; it needs a human decision, not more audio"
        )
    else:
        finding, detail = "inconclusive", (
            f"{b} regressed vs {c} recovered (exact p={p_value}) — directional but "
            f"underpowered; certifying at this rate would take {needed} paired items"
        )

    # ``finding`` says what we OBSERVED; ``certified`` says whether the set was big enough for
    # that observation to mean anything. They are not the same claim, and conflating them is
    # how "no_evidence_of_regression" gets read as "passes non-inferiority". Zero regressions
    # over 42 items still leaves an 8% upper bound — above the 5% margin — so it certifies
    # nothing. Non-inferiority is a statement about the bound, never the point estimate.
    #
    # Certification is taken on the PAIRED NET DIFFERENCE (b-c)/n (ADR-0006). The rule this
    # replaced also required ``b <= c``, which was not conservatism but a zero-tolerance rule
    # in disguise: ``c`` counts items base got wrong and the fine-tune got right, so whenever
    # base is clean on the set — it is, 0 silent corrections in 41 — ``c`` is pinned at 0 and
    # ``b <= c`` degenerates to ``b == 0``. No quantity of additional audio could ever satisfy
    # it, because concordant items move neither count. Dropping it is a real loosening and is
    # recorded as one; on the sets scored to date it changes no verdict, since with ``c == 0``
    # the net-difference bound equals the Wilson bound on ``b / n``.
    net_low, net_high = paired_score_interval(b, c, len(shared)) if shared else (None, None)
    certified = bool(shared) and net_high is not None and net_high <= max_regression

    return {
        "paired_items": len(shared),
        "regressed": b,
        "regressed_items": regressed,
        "recovered": c,
        "regression_rate": round(rate, 4) if rate is not None else None,
        "regression_upper95": round(high, 4) if high is not None else None,
        "net_difference": round((b - c) / len(shared), 4) if shared else None,
        "net_difference_ci95": (
            [round(net_low, 4), round(net_high, 4)] if net_high is not None else None
        ),
        "mcnemar_exact_p": round(p_value, 4),
        "max_regression": max_regression,
        "observed_direction": "tied" if b == c else ("worse" if b > c else "better"),
        "equality_finding": finding,
        "non_inferiority_certified": certified,
        # What it would take to settle this set, at the discordance rate it actually shows.
        # ``None`` means no amount of recording can — see ``required_items``.
        "items_needed_at_observed_rate": needed,
        "detail": detail,
    }


def verdict(summary: dict, max_silent_correction: float = MAX_SILENT_CORRECTION_RATE) -> dict:
    """Is the model fit to flag a student's vowel error?

    Deliberately NOT "does it follow the audio more often than not". Muraja is a recitation
    checker, so the decisive quantity is the **silent-correction rate**: how often a
    deliberately wrong vowel is transcribed as the canonical one. Every such case is an error
    the student is never told about. A model could follow the audio on 60% of items — clearing
    any coin-flip bar comfortably — and still be unfit.

    Judged against the UPPER 95% bound, not the point estimate, so a handful of items cannot
    buy a pass. Note that a low rate here is necessary but not sufficient: it is measured on
    deliberate, clearly-articulated errors from a single voice.
    """
    if summary["scored"] < 20:
        return {
            "conclusive": False,
            "reason": f"only {summary['scored']} scorable items — too few to rule",
        }
    low, high = summary["silent_correction_ci95"]
    tolerance = max_silent_correction
    if high <= tolerance:
        return {
            "conclusive": True, "fit_to_flag_vowel_errors": True,
            "tolerance": tolerance,
            "interpretation": f"silent-correction rate is at most {high:.1%} with 95% "
                              f"confidence, within the {tolerance:.0%} tolerance — on this "
                              "voice the model flags rather than silently corrects vowel errors",
        }
    if low > tolerance:
        return {
            "conclusive": True, "fit_to_flag_vowel_errors": False,
            "tolerance": tolerance,
            "interpretation": f"silent-correction rate is at least {low:.1%} with 95% "
                              f"confidence, above the {tolerance:.0%} tolerance — the model "
                              "reconstructs the canonical vowel too often to be trusted to "
                              "flag a student's error",
        }
    return {
        "conclusive": False,
        "tolerance": tolerance,
        "reason": f"95% CI on the silent-correction rate [{low}, {high}] spans the "
                  f"{tolerance:.0%} tolerance — more items needed to rule",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--items", type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--model")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--max-silent-correction", type=float, default=MAX_SILENT_CORRECTION_RATE,
        help="ceiling on the rate at which a wrong vowel is transcribed as the canonical one",
    )
    parser.add_argument(
        "--baseline", type=Path,
        help="a report from the base model; enables the ADR-0003 non-inferiority gate",
    )
    parser.add_argument("--max-regression", type=float, default=MAX_REGRESSION)
    parser.add_argument(
        "--rescore", type=Path,
        help="re-judge an existing report's stored per-item outcomes under the current rule, "
             "instead of decoding audio — no model or GPU needed",
    )
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()

    if args.rescore:
        existing = json.loads(args.rescore.read_text(encoding="utf-8"))
        results, model = existing["items"], existing.get("model")
    else:
        required = {"--items": args.items, "--manifest": args.manifest,
                    "--audio-dir": args.audio_dir, "--model": args.model}
        absent = [flag for flag, value in required.items() if value is None]
        if absent:
            raise SystemExit(f"{', '.join(absent)} are required unless --rescore is given")
        items = [json.loads(line)
                 for line in args.items.read_text(encoding="utf-8").splitlines() if line.strip()]
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
        model = args.model

    report = {
        "model": model,
        "summary": summarize(results),
        "items": results,
    }
    report["verdict"] = verdict(report["summary"], args.max_silent_correction)
    if args.baseline:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        report["vs_baseline"] = compare_to_baseline(
            report["items"], baseline["items"], args.max_regression
        )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "verdict": report["verdict"]},
                     indent=2, ensure_ascii=False))
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()

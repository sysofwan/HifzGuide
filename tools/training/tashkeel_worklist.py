"""Mine a tashkeel audit worklist from the Tadabur corpus (#60).

The P3.5 poison audit built its ``should_accept`` / ``should_reject`` sets by mining the
corpus for soft-pair and shadda contrasts — but that pipeline runs through
:func:`tadabur.normalization.normalize_phonemes`, which deletes ``U+064E``/``U+064F``/
``U+0650`` unconditionally. It is **blind to vowels by construction**, which is why the
fixture sets carry buckets for ``ذ↔ز``/``س↔ص``/``shadda`` and none for tashkeel.

:mod:`training.tashkeel_eval` is not blind — it anchors vowels to carriers on the raw
strings — and it already runs over thousands of held-out Tadabur windows. It reports that
the base checkpoint *omits* far more reference vowels than the fine-tune does. That
aggregate is the reason to care, but it cannot settle the question on its own: when the
base model declines to mark a vowel, the reference says nothing about whether the reciter
actually said it. Tadabur has no ground truth for the vowel a reciter produced.

So this module mines the positions where the two checkpoints **disagree** and hands them to
a human ear (:mod:`tadabur.tashkeel_audit_ui`). Concordant sites are deliberately excluded:
the comparison this feeds is McNemar's paired test, which is a function of the discordant
cells alone (:func:`tadabur.tashkeel_acceptance.compare`), so adjudicating sites both models
got right would cost audit hours and move no number.

Both directions are mined, in the same file and the same random order, because mining only
the flattering one would build a set that can only show the fine-tune winning:

* **recovered** — base failed the reference vowel, candidate matched it.
* **regressed** — base matched it, candidate failed.

The audit question is deliberately *not* "was the model right". It is "what did the reciter
say", which a listener can answer without knowing — and must not be told — which checkpoint
produced which outcome.

Runs on Linux + CUDA (see ``tools/README.md``).

Usage:
  python -m training.tashkeel_worklist \\
      --labels audit_run/seg_v21/windowed_labels_v2.jsonl \\
      --audio-dir audit_run/clips_v2 \\
      --candidate audit_run/seg_v21/rung3_v2/merged \\
      --out audit_run/seg_v21/tashkeel_worklist.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from collections.abc import Mapping
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from training.tashkeel_eval import (
    DAMMA,
    FATHA,
    KASRA,
    MATCHED,
    SHORT_VOWELS,
    VowelSite,
    _decode_windows,
    _load_windows,
    coverage_of,
    vowel_sites,
)

#: Human-readable names for the three colours, for the audit UI and per-bucket sampling.
VOWEL_NAMES = {FATHA: "fatha", DAMMA: "damma", KASRA: "kasra"}

#: The candidate improved on the base at this position.
RECOVERED = "recovered"
#: The candidate lost a position the base model had right.
REGRESSED = "regressed"
DIRECTIONS = (RECOVERED, REGRESSED)

#: The frozen base checkpoint failed this position. A superset of :data:`RECOVERED` for
#: *every* candidate, present and future — which is what makes it labellable in advance.
BASE_FAILED = "base_failed"
#: The base checkpoint matched it. Likewise a superset of :data:`REGRESSED`.
BASE_MATCHED = "base_matched"
STATIC_STRATA = (BASE_FAILED, BASE_MATCHED)

#: Sites drawn per (stratum, vowel) bucket by default. Six buckets, so ~300 sites — enough
#: to move a McNemar comparison well past the counterfactual set's 41 items while staying a
#: single sitting's work.
DEFAULT_PER_BUCKET = 50

#: Static mining draws unevenly on purpose. ``base_failed`` is where a candidate's recoveries
#: live and is ~88% recovered for a good fine-tune, so nearly every site labelled there pays
#: off. ``base_matched`` is where regressions live at ~2%, so it earns a much smaller draw:
#: it exists to keep the estimator unbiased rather than to measure regressions precisely,
#: which is the per-run top-up's job (and ADR-0006's).
STATIC_PER_BUCKET = {BASE_FAILED: 100, BASE_MATCHED: 50}


@dataclass(frozen=True)
class TashkeelSite:
    """One discordant vowel position, ready to be adjudicated by ear.

    ``start_sample`` / ``num_samples`` are the window's clip-relative span — the exact audio
    both checkpoints were fed — so the UI plays what the models heard rather than the whole
    clip. ``reference`` is the window's raw (un-normalized) phoneme label and
    ``reference_index`` points at the vowel inside it, so the page can show the site in
    context without this module inventing a display format.

    ``direction`` and the two ``*_outcome`` fields are what the comparison needs and what the
    listener must never see; :mod:`tadabur.tashkeel_audit_ui` withholds them.

    ``direction`` names which side of the **base** outcome this site sits on. Mined against a
    candidate it is :data:`RECOVERED` / :data:`REGRESSED`; mined statically, with no candidate
    in existence yet, it is :data:`BASE_FAILED` / :data:`BASE_MATCHED` and the two ``candidate_*``
    fields are empty — a candidate's outcome at these sites is supplied later by
    :mod:`training.tashkeel_outcomes` without any further listening.
    """

    site_id: str
    clip_audio_filename: str
    surah_ayah: str
    reciter_id: int
    window_index: int
    start_sample: int
    num_samples: int
    reference: str
    reference_index: int
    reference_vowel: str
    vowel_name: str
    carrier: str | None
    direction: str
    base_outcome: str
    candidate_outcome: str
    base_vowel: str | None
    candidate_vowel: str | None


SCHEMA_FIELDS = tuple(f.name for f in fields(TashkeelSite))


def site_id(clip_audio_filename: str, window_index: int, reference_index: int) -> str:
    """A stable id for one audited position, independent of sampling or file order.

    Adjudications are stored against this, so re-mining with a different seed, bucket size
    or candidate checkpoint resumes an audit already done rather than orphaning it.
    """
    key = f"{clip_audio_filename}#{window_index}@{reference_index}"
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _reference_sites(sites: list[VowelSite]) -> dict[int, VowelSite]:
    """Sites that classify a *reference* vowel, keyed by its index in the reference.

    Spurious decode vowels are dropped: they have no reference position to pair on, and the
    question this audit asks ("did the reciter say the reference vowel here?") is not
    defined for them.
    """
    return {
        site.reference_index: site
        for site in sites
        if site.reference_vowel is not None and site.reference_index is not None
    }


def discordant_sites(
    reference: str,
    base_decode: str,
    candidate_decode: str,
    label,
) -> list[TashkeelSite]:
    """Positions in one window where exactly one of the two checkpoints matched.

    Each checkpoint is aligned to the reference independently — they are different models
    and will trim different spans — and the results are paired on the **reference index**,
    which is the only coordinate both alignments share.
    """
    base = _reference_sites(vowel_sites(base_decode, reference))
    candidate = _reference_sites(vowel_sites(candidate_decode, reference))

    rows: list[TashkeelSite] = []
    for index in sorted(base.keys() & candidate.keys()):
        base_site, candidate_site = base[index], candidate[index]
        base_ok = base_site.outcome == MATCHED
        candidate_ok = candidate_site.outcome == MATCHED
        if base_ok == candidate_ok:
            continue
        direction = RECOVERED if candidate_ok else REGRESSED
        rows.append(
            TashkeelSite(
                site_id=site_id(label.clip_audio_filename, label.window_index, index),
                clip_audio_filename=label.clip_audio_filename,
                surah_ayah=label.surah_ayah,
                reciter_id=label.reciter_id,
                window_index=label.window_index,
                start_sample=label.start_sample,
                num_samples=label.num_samples,
                reference=reference,
                reference_index=index,
                reference_vowel=base_site.reference_vowel,
                vowel_name=VOWEL_NAMES[base_site.reference_vowel],
                carrier=base_site.carrier,
                direction=direction,
                base_outcome=base_site.outcome,
                candidate_outcome=candidate_site.outcome,
                base_vowel=base_site.decoded_vowel,
                candidate_vowel=candidate_site.decoded_vowel,
            )
        )
    return rows


def static_sites(reference: str, base_decode: str, label) -> list[TashkeelSite]:
    """Every reference vowel in one window, stratified by the **frozen base** outcome.

    The candidate-free half of the audit. :func:`discordant_sites` cannot run until a
    checkpoint exists, so its worklist — and therefore the listening — is pinned behind every
    training run. But the verdict a listener gives is *"the reciter said fatha"*: a fact about
    the audio, with no model in it. Only the *selection* was ever coupled.

    The base checkpoint is frozen, so ``base_failed`` is a permanent partition of the corpus,
    and it contains every recovery any future candidate can make. Labelling it in advance is
    therefore not a guess about the next fine-tune — it is the whole gain side, banked. This
    mirrors ``should_accept.jsonl``, which the base model helped *find* and which every rung
    since has been scored against without re-auditing a clip.

    The efficiency is asymmetric and the caller should know it. Base fails ~15.9% of reference
    vowels and a good fine-tune ~1.9%, so at least ``7145 - 839`` of the base-failed sites —
    88% — are recoveries for such a candidate, while at most ``839 / 37796`` = 2.2% of
    base-matched sites are regressions. The gain side banks almost perfectly; the cost side
    does not bank at all, and is topped up per run by :func:`discordant_sites`.
    """
    rows: list[TashkeelSite] = []
    for index, site in sorted(_reference_sites(vowel_sites(base_decode, reference)).items()):
        matched = site.outcome == MATCHED
        rows.append(
            TashkeelSite(
                site_id=site_id(label.clip_audio_filename, label.window_index, index),
                clip_audio_filename=label.clip_audio_filename,
                surah_ayah=label.surah_ayah,
                reciter_id=label.reciter_id,
                window_index=label.window_index,
                start_sample=label.start_sample,
                num_samples=label.num_samples,
                reference=reference,
                reference_index=index,
                reference_vowel=site.reference_vowel,
                vowel_name=VOWEL_NAMES[site.reference_vowel],
                carrier=site.carrier,
                direction=BASE_MATCHED if matched else BASE_FAILED,
                base_outcome=site.outcome,
                candidate_outcome="",
                base_vowel=site.decoded_vowel,
                candidate_vowel=None,
            )
        )
    return rows


def population_counts(
    references: list[str], rows: list[TashkeelSite], strata_names: tuple[str, ...] = DIRECTIONS
) -> dict:
    """Reference-vowel totals the sampled worklist must be read against.

    A worklist is a *sample* of the discordant positions. Reporting a paired result without
    these makes the audit look like a census of the corpus: the same 4 regressions mean one
    thing in 50 discordant sites and another in 5,000. :func:`tadabur.tashkeel_acceptance`
    reads ``reference_vowels`` as the denominator both rates are expressed over.

    ``strata`` counts each ``(direction, colour)`` cell separately, because that is the unit
    :func:`sample_worklist` draws on. Scaling a direction's *pooled* audited share onto its
    total would weight the colours by sample size rather than population size — with equal
    draws from a bucket of 10,000 fatha and one of 100 kasra, a colour-dependent
    confirmation rate makes the pooled estimate wrong by orders of magnitude.

    Derived from the mined ``rows`` rather than re-aligned, so the population a result is
    scaled onto and the sites offered for audit are guaranteed to be the same partition.
    ``reference_vowels`` needs no alignment at all: every short vowel in every reference is
    classified by exactly one site (:func:`training.tashkeel_eval.vowel_sites` covers the
    unaligned ends too), so counting the characters is counting the positions.
    """
    strata: dict[str, dict[str, int]] = {
        name: {vowel: 0 for vowel in VOWEL_NAMES.values()} for name in strata_names
    }
    for row in rows:
        strata[row.direction][row.vowel_name] += 1
    totals = {
        "reference_vowels": sum(1 for ref in references for c in ref if c in SHORT_VOWELS),
        "strata": strata,
        **{name: sum(strata[name].values()) for name in strata_names},
    }
    # Static mining partitions every reference vowel, so nothing is left over; paired mining
    # keeps only the discordant cells and the rest are positions both checkpoints agreed on.
    totals["concordant"] = totals["reference_vowels"] - sum(
        totals[name] for name in strata_names
    )
    return totals


def sample_worklist(
    rows: list[TashkeelSite], per_bucket: int | Mapping[str, int], seed: object = 0
) -> list[TashkeelSite]:
    """Up to ``per_bucket`` sites per (direction, vowel), drawn reproducibly.

    ``per_bucket`` may be a mapping from stratum to cap, because static mining wants an
    uneven draw: ``base_failed`` sites are ~88% recoveries for a good candidate and
    ``base_matched`` ~2% regressions, so a flat cap would spend half the listening on the
    stratum that yields almost nothing (see :data:`STATIC_PER_BUCKET`).

    Bucketing by colour as well as direction stops the draw being swallowed by fatha, which
    outnumbers kasra and damma together in the corpus; ADR-0003's collapse check is
    per-colour, so the audit has to be able to speak per-colour too.

    The result is shuffled across buckets, because a listener working through fifty
    consecutive "candidate recovered this" sites will infer the direction from the run
    length alone — which is exactly the knowledge the blind audit exists to withhold.

    The draw is **consistent under a changing population**: sites are ranked by a hash of
    their id and the top ``per_bucket`` taken, rather than sampled positionally. Both are
    uniform, but only the former re-draws the *same* sites when the bucket changes. That
    matters because every training run re-mines against a new candidate, and adjudications
    key on :func:`site_id`, so overlap between successive worklists is audit hours that do
    not have to be spent twice. Positional sampling gets no overlap worth having: fifty
    drawn from ~4,300 twice intersect in about one site even when the populations are 95%
    identical, so each run would restart the audit from nothing.
    """
    buckets: dict[tuple[str, str], list[TashkeelSite]] = {}
    for row in sorted(rows, key=lambda r: r.site_id):
        buckets.setdefault((row.direction, row.vowel_name), []).append(row)

    def rank(key: tuple[str, str], row: TashkeelSite) -> str:
        return hashlib.sha256(f"{seed}:{key}:{row.site_id}".encode("utf-8")).hexdigest()

    drawn: list[TashkeelSite] = []
    for key in sorted(buckets):
        bucket = buckets[key]
        cap = per_bucket if isinstance(per_bucket, int) else per_bucket[key[0]]
        if len(bucket) <= cap:
            drawn.extend(bucket)
        else:
            drawn.extend(sorted(bucket, key=lambda r: rank(key, r))[:cap])
    random.Random(f"{seed}:order").shuffle(drawn)
    return drawn


def write_worklist(path: Path, rows: list[TashkeelSite]) -> None:
    """Write the worklist as one JSON object per line, UTF-8, Arabic left readable."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(asdict(row), ensure_ascii=False) + "\n")


def read_worklist(path: Path) -> list[TashkeelSite]:
    """Read a worklist back, rejecting any row that is not exactly this schema.

    Strict because the audit UI and the paired comparison both key off these fields: a
    silently-missing ``direction`` would not crash, it would quietly drop sites from one arm
    of the comparison.
    """
    rows: list[TashkeelSite] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        record = json.loads(line)
        missing = set(SCHEMA_FIELDS) - record.keys()
        unknown = record.keys() - set(SCHEMA_FIELDS)
        if missing or unknown:
            raise ValueError(
                f"{path}:{number} does not match the worklist schema "
                f"(missing: {sorted(missing)}, unknown: {sorted(unknown)})."
            )
        rows.append(TashkeelSite(**record))
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True,
                        help="windowed CTC labels JSONL (training.windowed_labels).")
    parser.add_argument("--audio-dir", type=Path, required=True,
                        help="staged 16 kHz clip directory.")
    parser.add_argument("--candidate", default=None,
                        help="fine-tuned checkpoint (merged model dir or hub id). Omit to "
                             "mine the candidate-free static set, stratified on the frozen "
                             "base outcome, which can be labelled before a candidate exists.")
    parser.add_argument("--base", default="obadx/muaalem-model-v3_2",
                        help="base checkpoint the candidate is compared against.")
    parser.add_argument("--split", default="val",
                        help="label split to mine (default: the held-out val split).")
    parser.add_argument("--limit", type=int, default=0,
                        help="decode at most this many windows, sampled deterministically "
                             "across the whole split (default 0 = all of it).")
    parser.add_argument("--per-bucket", type=int, default=None,
                        help=f"sites to draw per (stratum, vowel) bucket (paired default "
                             f"{DEFAULT_PER_BUCKET}; static default {STATIC_PER_BUCKET}).")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", type=Path, required=True,
                        help="worklist JSONL; a '.summary.json' sidecar is written beside it.")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    labels = _load_windows(args.labels, args.split, args.limit or None)
    references = [label.phoneme_label for label in labels]
    if not any(c in SHORT_VOWELS for ref in references for c in ref):
        raise ValueError(
            f"{args.labels} '{args.split}' labels contain no short vowels — they were built "
            "from the normalized reference (ADR-0003). Rebuild from raw_reference_phonemes; "
            "mining a tashkeel worklist from a vowel-free reference would yield nothing."
        )
    coverage = coverage_of(labels)
    print(
        f"Mining {coverage['windows']} '{args.split}' windows — "
        f"{coverage['reciters']} reciters, {coverage['clips']} clips, "
        f"{coverage['ayahs']} ayahs.",
        flush=True,
    )

    base = _decode_windows(args.base, labels, args.audio_dir, args.batch_size, args.device)

    rows: list[TashkeelSite] = []
    if args.candidate:
        candidate = _decode_windows(
            args.candidate, labels, args.audio_dir, args.batch_size, args.device
        )
        for label, reference, base_decode, candidate_decode in zip(
            labels, references, base, candidate
        ):
            rows.extend(discordant_sites(reference, base_decode, candidate_decode, label))
        strata_names, per_bucket = DIRECTIONS, args.per_bucket or DEFAULT_PER_BUCKET
    else:
        for label, reference, base_decode in zip(labels, references, base):
            rows.extend(static_sites(reference, base_decode, label))
        strata_names = STATIC_STRATA
        per_bucket = args.per_bucket or STATIC_PER_BUCKET

    drawn = sample_worklist(rows, per_bucket)
    write_worklist(args.out, drawn)

    summary = {
        "coverage": coverage,
        "base": args.base,
        "candidate": args.candidate,
        "mode": "paired" if args.candidate else "static",
        "population": population_counts(references, rows, strata_names),
        "sampled": len(drawn),
        "per_bucket": per_bucket,
    }
    summary_path = args.out.with_suffix(args.out.suffix + ".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote {len(drawn)} sites to {args.out} (summary: {summary_path})")


if __name__ == "__main__":
    main()

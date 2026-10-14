"""Does committing block 1 change the app outcome? A replay through Muraja's engine (#148).

Muraja's transcriber sends each full window's first second (block 0) as the **hop** the engine
keeps, and the rest as provisional **overlap**. Every check's alignment query is the confirmed
tail, then every pending hop, then the current overlap or preview, and every word the
alignment reaches is regraded (the code reading, with citations, is
:data:`training.commit_block_doc.CODE_READING`). The commit block therefore decides which
decode stands for each second once it is older than the current window, so it reaches grading.
This runner measures how far, by replaying a transcriber that commits block 1 instead
(``training.decoding``, "Another commit block") through Muraja's own engine, beside today's.

1. **Population.** Ship A's word sample with its whole-word verdicts
   (:func:`tadabur.word_sample.verified_words`) and the verified sites of Ship A's frame in the
   held-out half (:mod:`tadabur.ship_a_verified`), the two sets #148 names; the shared exclusion
   (:func:`tadabur.exclusions.current`) applies as everywhere.
2. **Check sequences.** ``b = 0`` is each item's committed check sequence
   (:data:`training.truth_baseline.CHECK_STORE`). ``b = 1`` is recorded in
   :data:`CHECK_B1_STORE` by the same decoder, featurisation, fingerprint and identity rules, one
   inference per scheduled check (:meth:`training.decoding.Decoder.app_checks_at`). The same
   inferences also cut the ``b = 0`` sequence, whose digest each item keeps
   (:data:`B0_DIGEST`); every run refuses an item whose digest is not that of its committed
   ``b = 0`` sequence (:func:`check_pairing`), so ``b = 1`` against ``b = 0`` compares two cuts
   of the same inferences and nothing else.
3. **Replay.** Each model at each block is an arm (``base/checks``, ``base/checks_b1``,
   ``h448/checks``, ``h448/checks_b1``), replayed under every scoring of
   :data:`training.muraja_policy.SCORINGS` (:func:`training.truth_baseline.app_outcomes`).
4. **Report.** :data:`REPORT_PATH`, rendered as :data:`DOC_PATH`
   (:mod:`training.commit_block_doc`): per arm and scoring, the word sample's false flags with
   their causes (:mod:`training.word_flags`) and the verified sites' flagged words per side
   (:func:`training.truth_scorer.score`, with the per-allowance view); the paired differences
   :func:`comparisons`; and every word or site whose word the app flags under one block only,
   with the grade the engine kept under each.

It measures only: it recommends no protocol and sets no threshold.

Usage (from ``tools/``)::

    # On the GPU box, with the staged clips (decodes only what the b = 1 cache lacks):
    python -m training.commit_block --audio-dir <clips> --decode-only
    # On a Mac with the Muraja harness built, torch-free, from the committed sequences:
    python -m training.commit_block
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from tadabur.exclusions import Exclusions, current
from tadabur.listening_session import adjudicated, read_verdicts
from tadabur.ship_a import HELD_OUT
from tadabur.ship_a_verified import SHIP_A, Design, load_designs
from tadabur.staged_audio import REGISTRY_PATH, load_staged_clips
from tadabur.truth_sites import TruthSite
from tadabur.word_sample import VerifiedWord, load_words, verified_words
from training.commit_block_doc import render
from training.decoding import DEPLOYED_BLOCK, Decoder, app_checks_mode, committed_text
from training.muraja_policy import WORD_FLAGGED, SiteGrade, require_harness
from training.site_outcomes import side
from training.tashkeel_eval import write_text_atomically
from training.truth_baseline import (
    BASELINE_DIR,
    CHECKS,
    DEFAULT_MODELS,
    KeptGrades,
    Store,
    app_outcomes,
    arm,
    items_of,
    load_sites,
    model_caches,
    truth_site_files,
    update_cache,
)
from training.truth_scorer import MurajaGrades, score, scored_sites
from training.word_flags import false_flag_difference, flag, log_description, word_report

COMMIT_BLOCK_DIR = BASELINE_DIR.parent / "commit_block"
REPORT_PATH = COMMIT_BLOCK_DIR / "report.json"
DOC_PATH = Path(__file__).parent.parent.parent / "docs" / "commit-block.md"

#: The block the hypothetical transcriber commits.
BLOCK = 1
#: The arm protocol of its check sequence, beside the truth baseline's ``checks`` (b = 0).
CHECKS_B1 = f"{CHECKS}_b{BLOCK}"
#: Each b = 1 item's record of the b = 0 sequence cut from the same inferences.
B0_DIGEST = f"{CHECKS}_sha256"


def sequence_digest(checks: Sequence[dict]) -> str:
    """The SHA-256 of a check sequence as canonical JSON."""
    text = json.dumps(list(checks), ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _b1_sequence(decoder: Decoder, span: np.ndarray) -> dict:
    by_block = decoder.app_checks_at(span, (DEPLOYED_BLOCK, BLOCK))
    return {CHECKS_B1: by_block[BLOCK], B0_DIGEST: sequence_digest(by_block[DEPLOYED_BLOCK])}


#: Each item's check sequence for a transcriber committing :data:`BLOCK`, with the digest of
#: the b = 0 sequence the same inferences cut.
CHECK_B1_STORE = Store(COMMIT_BLOCK_DIR / CHECKS_B1, {CHECKS_B1: app_checks_mode(BLOCK)},
                       _b1_sequence, one_item_per_line=True)


def check_pairing(b0: Mapping[str, dict], b1: Mapping[str, dict], keys: Sequence[str]) -> None:
    """Every item's b = 1 sequence must come from the inferences that cut its committed b = 0
    sequence (module docstring)."""
    unpaired = [k for k in keys if b1[k][B0_DIGEST] != sequence_digest(b0[k][CHECKS])]
    if unpaired:
        raise ValueError(f"{len(unpaired)} item(s) were not re-decoded as committed at b = 0, "
                         f"e.g. {unpaired[0]}; the b = 1 sequence is not paired with it")


def held_out_ship_a(sites: Sequence[TruthSite], designs: Mapping[str, Design]) -> list[TruthSite]:
    """The verified sites of Ship A's frame in the held-out half."""
    return [s for s in sites if s.site_id in designs
            and (designs[s.site_id].frame, designs[s.site_id].half) == (SHIP_A, HELD_OUT)]


def comparisons(models: Sequence[str]) -> list[tuple[str, str]]:
    """The paired differences reported: each model's b = 1 against its b = 0, and h448 − base at
    each block."""
    pairs = [(arm(m, CHECKS_B1), arm(m, CHECKS)) for m in models]
    if {"base", "h448"} <= set(models):
        pairs += [(arm("h448", p), arm("base", p)) for p in (CHECKS, CHECKS_B1)]
    return pairs


def grade_view(grade: SiteGrade, text: str | None) -> dict:
    """What the app shows for one word under one arm and scoring, and why when flagged (a
    flagged word is placed, so ``text``, its phonemes on Muraja's reference, is known)."""
    return {
        "outcome": grade.outcome,
        "quality": grade.quality,
        "causes": list(flag(grade, text).causes) if grade.outcome in WORD_FLAGGED else [],
        "errors": [log_description(e) for e in grade.errors],
    }


def flips(described: Mapping[str, dict], kept: KeptGrades, models: Sequence[str]) -> dict:
    """Per model and scoring, every id of ``described`` whose word the app flags at one block
    only, with its description and the grade kept at each block."""
    out: dict[str, dict[str, list[dict]]] = {}
    for model in models:
        out[model] = {}
        for scoring, by_arm in kept.grades.items():
            b0, b1 = by_arm[arm(model, CHECKS)], by_arm[arm(model, CHECKS_B1)]
            out[model][scoring] = [
                {"id": i, **about, "b0": grade_view(b0[i], kept.texts.get(i)),
                 "b1": grade_view(b1[i], kept.texts.get(i))}
                for i, about in described.items()
                if (b0[i].outcome in WORD_FLAGGED) != (b1[i].outcome in WORD_FLAGGED)
            ]
    return out


def word_section(words: Sequence[VerifiedWord], kept: KeptGrades, models: Sequence[str],
                 excluded: Exclusions) -> dict:
    """The word sample's report (:func:`training.word_flags.word_report`), with the paired
    false-flag differences and the words that flip."""
    report = word_report(words, kept.grades, kept.texts, excluded)
    report["differences"] = {
        f"{a} - {b}": {scoring: false_flag_difference(words, by_arm[a], by_arm[b], excluded)
                       for scoring, by_arm in kept.grades.items()}
        for a, b in comparisons(models)
    }
    judged = {w.site.site_id: {} for w in words if w.site.site_id not in excluded and w.side}
    report["flips"] = flips(judged, kept, models)
    return report


def site_section(sites: Sequence[TruthSite], reciter_of: Mapping[str, int],
                 decodes: Mapping[str, Mapping[str, str]], muraja: MurajaGrades, kept: KeptGrades,
                 designs: Mapping[str, Design], models: Sequence[str], excluded: Exclusions) -> dict:
    """The held-out Ship A sites: :func:`training.truth_scorer.score` over them alone (its pooled
    cells and per-allowance view), and the sites that flip."""
    weights = {site_id: d.weight for site_id, d in designs.items()}
    report = score(sites, reciter_of, decodes, comparisons(models), muraja,
                   design_weights=weights, excluded_items=excluded)
    rows = scored_sites(sites, reciter_of, decodes, design_weights=weights,
                        excluded_items=excluded).rows
    described = {row.site.site_id: {"side": side(row.site), "mark": row.site.mark,
                                    "heard": row.site.heard, "prescribed": row.site.prescribed,
                                    "stratum": row.site.stratum}
                 for row in sorted(rows, key=lambda r: r.site.site_id)}
    return {
        "sites": report["sites"],
        "cells": [c for c in report["cells"] if c["family"] == "all" and c["sites"]],
        "allowances": report["allowances"],
        "flips": flips(described, kept, models),
    }


def transcripts(sequences: Mapping[str, Mapping[str, Sequence[dict]]], models: Sequence[str]) -> dict:
    """Per model, how the two blocks' sequences differ: checks sent and items whose committed
    transcript differs."""
    out = {}
    for model in models:
        b0, b1 = sequences[arm(model, CHECKS)], sequences[arm(model, CHECKS_B1)]
        out[model] = {
            "items": len(b0),
            "checks": {CHECKS: sum(map(len, b0.values())), CHECKS_B1: sum(map(len, b1.values()))},
            "different_transcripts": sum(committed_text(b0[k]) != committed_text(b1[k]) for k in b0),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", type=Path, help="staged 16 kHz clips; needed only to decode")
    parser.add_argument("--decode-only", action="store_true",
                        help="fill the b = 1 cache and stop: the GPU box has no Muraja harness")
    args = parser.parse_args()
    models = DEFAULT_MODELS

    verdicts = read_verdicts()
    designs = load_designs()
    sites = held_out_ship_a(adjudicated(load_sites(truth_site_files()), verdicts), designs)
    words = verified_words(load_words(), verdicts)
    registry = load_staged_clips(REGISTRY_PATH)
    items = items_of([*sites, *(word.site for word in words)], registry)
    print(f"{len(sites)} held-out Ship A sites and {len(words)} sampled words on {len(items)} items",
          flush=True)

    sequences: dict[str, dict[str, list[dict]]] = {}
    fingerprints: dict[str, dict] = {}
    for name, ref in models.items():
        _, b0 = model_caches(name, ref, items, registry, None)
        b1 = update_cache(CHECK_B1_STORE, name, ref, items, registry, args.audio_dir)
        if b1["identity"] != b0["identity"]:
            raise ValueError(f"{name}: its b = 0 and b = 1 sequences come from different weights")
        check_pairing(b0["items"], b1["items"], sorted(items))
        fingerprints[name] = {**b0["fingerprints"], **b1["fingerprints"]}
        sequences[arm(name, CHECKS)] = {k: b0["items"][k][CHECKS] for k in items}
        sequences[arm(name, CHECKS_B1)] = {k: b1["items"][k][CHECKS_B1] for k in items}
    if args.decode_only:
        print(f"b = {BLOCK} sequences complete and paired for {sorted(models)}; grade them where "
              "the Muraja harness is built")
        return

    decodes = {a: {k: committed_text(c) for k, c in by_key.items()} for a, by_key in sequences.items()}
    excluded = current()
    reciter_of = {name: clip.reciter_id for name, clip in registry.items()}
    muraja, kept = app_outcomes(sites, words, decodes, sequences, require_harness())
    report = {
        "models": dict(models),
        "fingerprints": fingerprints,
        "block": BLOCK,
        "muraja": muraja.as_dict(),
        "transcripts": transcripts(sequences, list(models)),
        "comparisons": [f"{a} - {b}" for a, b in comparisons(list(models))],
        "word_sample": word_section(words, kept, list(models), excluded),
        "verified": site_section(sites, reciter_of, decodes, muraja, kept, designs, list(models),
                                 excluded),
    }
    write_text_atomically(REPORT_PATH, json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    write_text_atomically(DOC_PATH, render(report))
    print(f"Wrote {REPORT_PATH} and {DOC_PATH}")


if __name__ == "__main__":
    main()

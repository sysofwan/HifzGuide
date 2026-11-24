"""P2, the CoreML-vs-PyTorch bridge (#309): per-word strict outcome agreement, bar 97%.

The App arm runs the app's own CoreML decoder at the weights v1.0.7 ships; the PyTorch side runs
the committed round-3 candidate EMA decode of the same weights through the same engine. P2 asks
whether the two name the same words as flagged, passed or ungraded: if they do not, S7 cannot
read the Teacher-vs-Student-PT comparison as a model difference alone.

The comparison is over outcomes only — two flags for different causes agree, since the bridge is
about *whether* the word fell, not why. The two sides must name the same words: a bridge that
silently dropped the words one side lacks would read better than it is, so a differing word set
is refused.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

#: The agreement P2 must reach for S7 to read its comparison as the models' alone.
P2_BAR = 0.97


def outcome(cell: Mapping) -> str:
    """A word's strict outcome: ``flagged``, ``passed`` or ``ungraded``."""
    if not cell["graded"]:
        return "ungraded"
    return "flagged" if cell["flagged"] else "passed"


def agreement(app: Sequence[Mapping], pytorch: Sequence[Mapping]) -> dict:
    """How many words the two runs place in the same strict outcome, and every disagreement."""
    left = {(row["item"], row["word"]): outcome(row["strict"]) for row in app}
    right = {(row["item"], row["word"]): outcome(row["strict"]) for row in pytorch}
    if len(left) != len(app) or len(right) != len(pytorch):
        raise ValueError("a word appears twice in one side of the bridge")
    if set(left) != set(right):
        raise ValueError(f"the two sides name different words: {len(set(left) - set(right))} "
                         f"only in the App arm, {len(set(right) - set(left))} only in PyTorch")
    disagreements = [{"item": item, "word": word, "app": left[(item, word)],
                      "pytorch": right[(item, word)]}
                     for item, word in sorted(left) if left[(item, word)] != right[(item, word)]]
    words = len(left)
    agree = words - len(disagreements)
    return {"words": words, "agree": agree,
            "rate": agree / words if words else 1.0,
            "disagreements": disagreements}


def bridge_verdict(rate: float) -> str:
    """``met`` at or above :data:`P2_BAR`, ``below`` under it — a report, never an exception."""
    return "met" if rate >= P2_BAR else "below"


# --- the bridge over the committed decode -------------------------------------------------------

#: The committed round-3 candidate EMA PyTorch decode: the bridge's PyTorch side, pinned in
#: ``tadabur.tlog_mining.candidate_inventory.PINS`` and the weights v1.0.7 ships.
R3_SOURCE = "tlog_mining/decodes/tlog.gating.r3_candidate_ema.jsonl.gz"

#: The seeded clip subset the bar is read on, and the seed that draws it (the issue's 200 clips).
P2_CLIPS = 200
P2_SEED = 319


def p2_section(population, app_grades, *, harness,
               subset_size: int = P2_CLIPS, seed: int = P2_SEED,
               all_clips: bool = False, failed: Sequence[str] = ()) -> dict:
    """The P2 readout: the App arm against the committed PyTorch decode, word by word.

    The commit decision reuses the committed decode instead of spending GPU hours re-deriving
    the same weights: v1.0.7 ships the round-3 candidate EMA, so the two sides differ in the
    decoder (CoreML 6-bit against PyTorch), not in the weights. The bar is read on the seeded
    subset of the **graded** clips; ``all_clips`` adds the whole-population agreement as the
    secondary line. A clip the App run failed has no word table, so it is left out of the draw
    and listed in ``skipped`` instead of aborting the report.
    """
    from tadabur.followalong_audit.clips import draw_clips
    from tadabur.tlog_mining.candidate_inventory import GATING, R3, load_decodes
    from training.nonpro_decode import item_key

    checks = load_decodes()[GATING][R3]
    graded = {row["item"] for row in app_grades}
    clip_of = {item_key(clip["audio_path"]): clip for clip in population}
    available = [clip_of[key] for key in clip_of if key in graded]
    chosen = draw_clips(available, subset_size, seed=seed)
    keys = {item_key(clip["audio_path"]) for clip in chosen}
    subset = agreement([row for row in app_grades if row["item"] in keys],
                       grade_checked_clips(chosen, checks, harness))
    everything = (agreement(app_grades, grade_checked_clips(available, checks, harness))
                  if all_clips else None)
    return {"bar": P2_BAR, "verdict": bridge_verdict(subset["rate"]), "source": R3_SOURCE,
            "seed": seed, "subset_clips": len(chosen), "graded_clips": len(available),
            "skipped": sorted(set(failed) & set(clip_of)), "subset": subset, "all": everything}


def grade_checked_clips(clips, checks, harness) -> list[dict]:
    """Replay each clip's recorded checks through the engine and grade every word of its ayah."""
    from tadabur.followalong_audit.app_arm import causes_of
    from tadabur.followalong_audit.app_run import word_grade_rows
    from tadabur.followalong_audit.session import session_request
    from training.muraja_policy import AUDIT_SCORINGS, MurajaText, word_traces
    from training.nonpro_decode import item_key

    text = MurajaText(harness.quran_db)
    requests, shapes = [], []
    for clip in clips:
        key = item_key(clip["audio_path"])
        if key not in checks:
            raise ValueError(f"{key} has no committed checks in the PyTorch decode")
        surah, ayah = (int(part) for part in clip["filename_label"].split(":"))
        word_count = len(text.phonemes(surah, ayah).split(" "))
        requests.append(session_request(key, surah, ayah, word_count, checks[key]))
        shapes.append((surah, ayah, word_count))
    _, results = harness.run(requests)
    rows = []
    for request, (surah, ayah, word_count) in zip(requests, shapes):
        traces = word_traces(request, results, AUDIT_SCORINGS)
        rows.extend(word_grade_rows(request["item"], word_count, traces,
                                   causes_of(text, surah, ayah)))
    return rows

"""#267 step 4: score the committed FastConformer decodes and write the report and the doc.

One letter-level rule on every column and the same located, non-G6 sites: what the whole-item
text carries at the site's carrier (heard / prescribed / other / none; ``none`` is flagged),
whether the edit's label is heard and the decoy left unchanged, and unique catches. Correct and
catch sides are never pooled. Intervals are reciter-clustered bootstraps (#245's
:func:`training.zipformer_readout.rate_view`); the small catch sets cannot carry much.

The frozen GO / PRIOR / STOP rule is applied once, to the report's own rates. The doc it writes
says what the intervals cannot carry and that a GO means only worth trying.

Usage (from ``tools/``, after the decodes are committed)::

    python -m tadabur.fastconformer_screen.readout
"""

from __future__ import annotations

import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path

from tadabur.fastconformer_screen.mapping import (carried_at, glyphs, imlaei_words, letters_of,
                                                  locate_edit, locate_site)
from tadabur.fastconformer_screen.outcomes import (COLUMNS, HEARD, NONE, OUTCOMES, OTHER,
                                                   PRESCRIBED, breakdown, is_flagged, outcome,
                                                   student_carried, unique_catches)
from tadabur.fastconformer_screen.sets import (edit_items, located_ship_a_sites, split_sides,
                                               teacher_bars)
from tadabur.fastconformer_screen.verdict import GO, STOP, Rates, verdict

DECODES = Path(__file__).parent / "decodes"
CARD_LINE_PATH = Path(__file__).parent / "card_line.json"
REPORT_PATH = Path(__file__).parent / "report.json"
DOC_PATH = Path(__file__).resolve().parents[3] / "docs" / "fastconformer-screen.md"
STUDENTS = ("base", "c95", "r2_candidate_ema")


def decode_lines(set_name: str, directory: Path = DECODES) -> dict[str, dict]:
    """A committed FastConformer decode set by item key (``.jsonl.gz`` or ``.jsonl``)."""
    from training.zipformer_readout import read_lines

    for suffix in (".fastconformer.jsonl.gz", ".fastconformer.jsonl"):
        path = directory / f"{set_name}{suffix}"
        if path.is_file():
            return read_lines(path)
    raise FileNotFoundError(f"{directory}: no {set_name} FastConformer decodes")


def _rate(observations: Mapping[str, Sequence[tuple[object, int, int]]],
          comparisons: Sequence[tuple[str, str]] = (), scale: float = 100.0) -> dict:
    from training.zipformer_readout import rate_view

    return rate_view(observations, comparisons, scale)


def _site_rows(subset, flagged, clusters):
    return sorted((clusters[site.site_id], int(flagged(site)), 1) for site in subset)


def ship_a_carried(sites, lines: Mapping[str, dict]) -> dict[str, str | None]:
    """FastConformer's carried value at each located site: its letter, or its shadda state."""
    from training.truth_scorer import item_key

    carried: dict[str, str | None] = {}
    for site in sites:
        located = locate_site(site)
        if located is None:
            continue
        line = lines[item_key(site)]
        carried[site.site_id] = carried_at(located.letters, located.index, line["ctc"],
                                           shaddah=site.mark == "shaddah")
    return carried


def _counts(subset, carried_by_arm: Mapping[str, Mapping[str, str | None]]) -> dict:
    """Per column: the flag rate, the outcome breakdown and the sensitivity line."""
    out = {}
    for arm in COLUMNS:
        carried = carried_by_arm[arm]
        outcomes = [outcome(carried[site.site_id], site.prescribed, site.heard)
                    for site in subset]
        out[arm] = {
            "n": len(subset),
            "flagged": sum(is_flagged(carried[site.site_id], site.prescribed) for site in subset),
            "rate": 100.0 * sum(is_flagged(carried[site.site_id], site.prescribed)
                                for site in subset) / len(subset),
            "breakdown": breakdown(outcomes),
            "sensitivity_flagged": sum(
                carried[site.site_id] is not None
                and carried[site.site_id] != site.prescribed for site in subset),
        }
    return out


def _edit_texts(set_name: str, lines: Mapping[str, dict]) -> dict[str, str]:
    return {key: line["ctc"] for key, line in lines.items()}


def fc_edit_carried(items, lines: Mapping[str, dict]) -> dict[str, str | None]:
    """FastConformer's carried value at each located edit: its Imlaei letter or shadda state."""
    carried: dict[str, str | None] = {}
    for item in items:
        located = locate_edit(item)
        if located is None:
            continue
        line = lines[item["item_id"]]
        carried[item["item_id"]] = carried_at(
            located.letters, located.index, line["ctc"],
            shaddah=item["operation"].startswith("shaddah"))
    return carried


def edit_rows(subset, agrees) -> dict[str, list[tuple[object, int, int]]]:
    """Each arm's bootstrap observations on one edit role, clustered by reciter.

    #267 asks for a reciter-clustered bootstrap, and the 150 pairs come from 44 reciters: a
    pair id would make every observation its own cluster and widen the interval.
    """
    return {arm: [(item["source"]["reciter_id"], int(agrees(arm, item)), 1)
                  for item in subset]
            for arm in ("fastconformer", *STUDENTS)}


def edit_scores(items) -> dict:
    """Each column's heard rate on the edits and unchanged rate on the decoys.

    FastConformer is read through the Imlaei mapping (its text is Imlaei); the students are
    read with ``teacher_agrees`` on their phonetic text, the rule the bars were computed with.
    """
    from tadabur.fastconformer_screen.outcomes import decoy_unchanged, edit_heard
    from tadabur.synthetic_edits import teacher_agrees
    from training.zipformer_readout import whole_item_texts

    lines = decode_lines("edits")
    fc_carried = fc_edit_carried(items, lines)
    texts = {model: whole_item_texts("edits")[model] for model in STUDENTS}
    # an unlocated edit is excluded, never counted as a miss, and every arm reads the same set
    located = [item for item in items if locate_edit(item) is not None]
    out = {}
    for role in ("edit", "decoy"):
        subset = [item for item in located if item["role"] == role]

        def agrees(arm, item):
            if arm == "fastconformer":
                return (edit_heard(fc_carried[item["item_id"]], item["label"])
                        if role == "edit"
                        else decoy_unchanged(fc_carried[item["item_id"]], item["prescribed"]))
            return teacher_agrees(item, texts[arm][item["item_id"]])

        observations = edit_rows(subset, agrees)
        view = _rate(observations, [("fastconformer", "base")])
        out[role] = {
            "n": len(subset),
            "counts": {arm: sum(agrees(arm, item) for item in subset) for arm in observations},
            "rates": view["models"], "differences": view["comparisons"],
        }
    return out


def word_disagreement(words: Sequence[str], hypothesis: str) -> tuple[int, int]:
    """``(disagreeing words, scored words)`` for one passage's reference and decode.

    A word disagrees when any of its Imlaei letters is substituted or deleted, its shadda state
    differs, or a glyph is inserted inside or beside it: an insertion belongs to the word it
    follows (a leading insertion to the first word), so every decoded glyph is attributed.
    """
    from tadabur.fastconformer_screen.mapping import align_ops

    reference = tuple(glyph for word in words for glyph in glyphs(word))
    if not reference:
        return 0, 0
    word_of = [index for index, word in enumerate(words) for _ in glyphs(word)]
    hypothesis_glyphs = glyphs(hypothesis)
    wrong = [False] * len(words)
    previous_word = None
    for kind, position, aligned in align_ops("".join(glyph.letter for glyph in reference),
                                             letters_of(hypothesis)):
        if position is None:
            wrong[previous_word if previous_word is not None else word_of[0]] = True
            continue
        previous_word = word_of[position]
        if kind != "match" or hypothesis_glyphs[aligned].shadda != reference[position].shadda:
            wrong[previous_word] = True
    return sum(wrong), len([word for word in words if glyphs(word)])


def qdat_disagreement(lines: Mapping[str, dict]) -> dict:
    """Word disagreement with the Imlaei reference per 1,000 words (consonant + shadda)."""
    from tadabur.nonpro_sources import QDAT
    from training.nonpro_eval import decodes_file

    import gzip

    with gzip.open(decodes_file(QDAT, "base"), "rt", encoding="utf-8") as handle:
        passages = {row["key"]: row for row in (json.loads(line) for line in handle if line)}

    observations = []
    for key, line in sorted(lines.items()):
        passage = passages[key]
        first, last = passage["word_from"], passage["word_to"]
        words = imlaei_words(passage["surah_ayah"], first - 1, last - first + 1)
        bad, total = word_disagreement(words, line["ctc"])
        observations.append((passage["reciter"], bad, total))
    view = _rate({"fastconformer": observations}, (), 1000.0)
    return {"clips": len(lines), "words": sum(row[2] for row in observations),
            "disagreeing_words": sum(row[1] for row in observations),
            "rule": "a reference word disagrees when any of its Imlaei letters is substituted, "
                    "deleted, inserted into or beside it, or aligned to a different shadda "
                    "state; harakat are never read",
            "rates": view["models"]}


def card_line_record(path: Path = CARD_LINE_PATH) -> dict:
    """Step 1's card sanity line, measured on the box and committed beside the decodes."""
    from tadabur.fastconformer_screen.gates import ScreenStop

    path = Path(path)
    if not path.is_file():
        raise ScreenStop(f"no card sanity line at {path}: step 1 measures it before the "
                         f"report, and the rule stops if it is above twice the card's WER")
    return json.loads(path.read_text(encoding="utf-8"))


def located_edits() -> list[dict]:
    """The edit items step 2 locates: the ones every arm is scored on."""
    return [item for item in edit_items() if locate_edit(item) is not None]


def mapping_record(located_sites: Sequence, edits: Sequence) -> dict:
    """Step 2's counts, its two denominators and every item it could not locate.

    The pre-registration says "sites", so the gate reads all of them (the held-out count
    included); the scored denominator is reported beside it, because that is the set the table
    is actually read on. Building the record does not gate, so a refusal can still list what
    it excluded.
    """
    from tadabur.fastconformer_screen.sets import ship_a_sites

    sites = ship_a_sites()
    all_mistakes, all_correct = split_sides(sites)
    located_mistakes, located_correct = split_sides(located_sites)
    denominators = {
        "sites": {"located": len(located_sites), "total": len(sites)},
        "scored": {"located": len(located_mistakes) + len(located_correct),
                   "total": len(all_mistakes) + len(all_correct)},
    }
    for entry in denominators.values():
        entry["share"] = (100.0 * entry["located"] / entry["total"]) if entry["total"] else 0.0
    located_ids = {site.site_id for site in located_sites}
    located_edit_ids = {item["item_id"] for item in edits}
    items = edit_items()
    return {
        "located": len(located_sites),
        "total": len(sites),
        # sites with no human verdict, so never scored: all of them, and the located ones
        "undecided": len(sites) - len(all_mistakes) - len(all_correct),
        "located_undecided": len(located_sites) - len(located_mistakes) - len(located_correct),
        "unlocated": sorted(site.site_id for site in sites if site.site_id not in located_ids),
        "edits": {"located": len(edits), "total": len(items),
                  "unlocated": sorted(item["item_id"] for item in items
                                      if item["item_id"] not in located_edit_ids)},
        "gates": denominators,
    }


def require_mapping_gate(mapping: Mapping) -> None:
    """Refuse below 95% on either denominator, or on the edits; nothing else gates here."""
    from tadabur.fastconformer_screen.gates import mapping_gate

    for entry in mapping["gates"].values():
        mapping_gate(entry["located"], entry["total"])
    mapping_gate(mapping["edits"]["located"], mapping["edits"]["total"])


def refusal_record(gate: str, message: str, **context) -> dict:
    """A gate's refusal as the report it leaves behind: the reason, and what it excluded."""
    return {"issue": 267, "verdict": STOP, "stopped_by": gate, "refusal": message, **context}


def refusal_doc(record: Mapping) -> str:
    """The doc of a stopped run: the verdict, the gate's message and the unlocated items."""
    lines = [
        "# FastConformer screen: stopped at a gate",
        "",
        f"**Verdict: STOP.** {record['refusal']}",
        "",
        "The pre-registered rule stops here. This is a refusal with a message, not a judgement",
        "of the model, and no screen result is reported.",
        "",
    ]
    mapping = record.get("mapping")
    if mapping is not None:
        lines += [f"{len(mapping['unlocated'])} Ship A sites and "
                  f"{len(mapping['edits']['unlocated'])} edits did not locate:", ""]
        lines += [f"- `{site_id}`" for site_id in mapping["unlocated"]]
        lines += [f"- `{item_id}`" for item_id in mapping["edits"]["unlocated"]]
        lines.append("")
    return "\n".join(lines)


def site_report() -> dict:
    """Every section of #267's step 4, then the frozen rule's verdict.

    Step 1's card gate is read first: a failed gate is a STOP record, and it must not need
    decodes that a failed runtime never produced. Step 2's mapping gate refuses with a message
    (``ScreenStop``); the report entry point writes that refusal down before exiting.
    """
    from tadabur.staged_audio import REGISTRY_PATH, load_staged_clips
    from training.truth_scorer import item_key

    card = card_line_record()
    if card["gate_failed"]:
        measured = card["measured"]
        return refusal_record(
            "card sanity",
            f"card sanity: CTC WER {measured['wer']:.2f} is above twice the card's "
            f"{card['card']['wer']:.2f}; the head, sample rate or normaliser is wrong",
            card_sanity=card)
    sites = located_ship_a_sites()
    edits = located_edits()
    mapping = mapping_record(sites, edits)
    require_mapping_gate(mapping)
    mistakes, correct = split_sides(sites)
    registry = load_staged_clips(REGISTRY_PATH)
    clusters = {site.site_id: registry[site.audio_filename].reciter_id for site in sites}

    lines = decode_lines("ship-a")
    missing = sorted({item_key(site) for site in sites} - set(lines))
    if missing:
        raise ValueError(f"ship-a: {len(missing)} located items have no decode, e.g. {missing[:3]}")
    fastconformer = ship_a_carried(sites, lines)
    carried = {"fastconformer": fastconformer}
    for model in STUDENTS:
        carried[model] = student_carried(model, sites)

    sections = {"mistakes": _counts(mistakes, carried), "correct": _counts(correct, carried)}
    views = {}
    for side, subset in (("mistakes", mistakes), ("correct", correct)):
        observations = {arm: _site_rows(subset, lambda site, arm=arm: is_flagged(
            carried[arm][site.site_id], site.prescribed), clusters) for arm in COLUMNS}
        views[side] = _rate(observations, [(arm, "base") for arm in COLUMNS if arm != "base"])

    outcomes = {site.site_id: {arm: outcome(carried[arm][site.site_id], site.prescribed, site.heard)
                               for arm in COLUMNS} for site in mistakes}
    caught = unique_catches(outcomes)

    edits_scores = edit_scores(edit_items())
    qdat = qdat_disagreement(decode_lines("qdat"))

    teacher = teacher_bars()
    fast = Rates(
        mistakes_flagged=sections["mistakes"]["fastconformer"]["rate"],
        correct_flagged=sections["correct"]["fastconformer"]["rate"],
        edits_heard=100.0 * edits_scores["edit"]["counts"]["fastconformer"]
        / edits_scores["edit"]["n"],
        decoys_unchanged=100.0 * edits_scores["decoy"]["counts"]["fastconformer"]
        / edits_scores["decoy"]["n"],
    )
    return {
        "issue": 267,
        "rule": "letter-level at each located site: heard / prescribed / other / none; "
                "flagged = not prescribed (none counts); correct and catch sides never pooled",
        "teacher": teacher.__dict__,
        "fastconformer": fast.__dict__,
        "sections": sections,
        "intervals": views,
        "edits": edits_scores,
        "qdat": qdat,
        "unique_catches": caught,
        "mapping": mapping,
        "card_sanity": card,
        "stopped_by": None,
        "run_note": {"device": "cpu", "gpu_hours": 0, "yielded_seconds": 0},
        "verdict": verdict(teacher, fast, len(caught)),
    }


def _pct(row: Mapping) -> str:
    interval = row.get("interval") or {}
    if "lower" in interval:
        return f"{row['point']:.1f}% [{interval['lower']:.1f}, {interval['upper']:.1f}]"
    return f"{row['point']:.1f}%"


def report_doc(record: Mapping) -> str:
    """The plain-language doc: summary and verdict first, then the evidence."""
    if record.get("refusal"):
        return refusal_doc(record)
    teacher = Rates(**record["teacher"])
    fast = Rates(**record["fastconformer"])
    section = record["sections"]
    caught = record["unique_catches"]
    lines = [
        "# FastConformer screen: a fourth mining detector on Ship A's verified sites",
        "",
        "## Summary",
        "",
        f"**Verdict: {record['verdict']}** (the frozen rule on the bars below; the owner's",
        "acknowledgement of the erratum is required before the verdict is read).",
        "",
        "NVIDIA's Arabic FastConformer was run on CPU as a detector only: its committed text was",
        "read at each Ship A site and at each synthetic edit, with one letter-level rule for every",
        "column. It is never a label, reference or training signal.",
        "",
        f"- Ship A mistakes flagged: FastConformer {fast.mistakes_flagged:.1f}%, teacher "
        f"{teacher.mistakes_flagged:.1f}% (the teacher's bar moved 5 points).",
        f"- Ship A correct sites flagged: FastConformer {fast.correct_flagged:.1f}%, teacher "
        f"{teacher.correct_flagged:.1f}% (the stop bar is the teacher + 5).",
        f"- Edits heard: FastConformer {fast.edits_heard:.1f}%, teacher {teacher.edits_heard:.1f}%; "
        f"decoys unchanged: FastConformer {fast.decoys_unchanged:.1f}%, teacher "
        f"{teacher.decoys_unchanged:.1f}%.",
        f"- Unique catches (a decisive mistake it carries and none of the three models does): "
        f"{len(caught)}.",
        "",
        "A GO means only **worth trying** in #261's scale-up draw. It does not rank this model",
        "above the three detectors, and it does not adopt it.",
        "",
        "## What was run",
        "",
        "- Model `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` at revision",
        "  `7f32349d952f42a28dce979ba73270aa2bbdfa89`; the `.nemo` SHA-256 was verified locally",
        "  against the full LFS object id (`d29d19d7...c3e31`).",
        "- Decoder: greedy CTC head, whole item, 16 kHz mono, on CPU. NeMo ran in its own",
        "  environment behind a 10 GB disk reserve; the environment and the weights were deleted",
        "  after the decodes were committed. No GPU time and no round-3 input.",
        f"- Card sanity line: CTC WER {record['card_sanity']['measured']['wer']:.2f} on "
        f"{record['card_sanity']['measured']['clips']} benchmark EveryAyah clips, the card's",
        f"  normalisation (`card_norm`); the card's CTC WER is "
        f"{record['card_sanity']['card']['wer']:.2f} and the run stops only above twice it.",
        f"  Measured on the benchmark's studio reciters, not NVIDIA's own test split.",
        f"- Sets: Ship A's verified sites minus the 8 G6 held-out sites (located, "
        f"{section['mistakes']['fastconformer']['n']} mistakes and "
        f"{section['correct']['fastconformer']['n']} correct sites), the "
        f"{record['mapping']['edits']['total']} synthetic edit/decoy items and qdat's "
        f"{record['qdat']['clips']:,} clips.",
        "",
        "## Bars",
        "",
        "| column | mistakes flagged | correct flagged | edits heard | decoys unchanged |",
        "|---|---:|---:|---:|---:|",
        f"| teacher | {teacher.mistakes_flagged:.1f}% | {teacher.correct_flagged:.1f}% | "
        f"{teacher.edits_heard:.1f}% | {teacher.decoys_unchanged:.1f}% |",
        f"| FastConformer | {fast.mistakes_flagged:.1f}% | {fast.correct_flagged:.1f}% | "
        f"{fast.edits_heard:.1f}% | {fast.decoys_unchanged:.1f}% |",
        "",
        "FastConformer's intervals (reciter-clustered bootstrap):",
        "",
        f"- mistakes flagged: {_pct(record['intervals']['mistakes']['models']['fastconformer'])}",
        f"- correct flagged: {_pct(record['intervals']['correct']['models']['fastconformer'])}",
        f"- edits heard: {_pct(record['edits']['edit']['rates']['fastconformer'])}",
        f"- decoys unchanged: {_pct(record['edits']['decoy']['rates']['fastconformer'])}",
        "",
        "Outcome breakdown (FastConformer):",
        "",
        "| side | heard | prescribed | other | none |",
        "|---|---:|---:|---:|---:|",
    ]
    for side in ("mistakes", "correct"):
        row = section[side]["fastconformer"]["breakdown"]
        lines.append(f"| {side} | {row['heard']} | {row['prescribed']} | {row['other']} | "
                     f"{row['none']} |")
    lines += [
        "",
        f"Measured text prior: at {section['mistakes']['fastconformer']['breakdown']['prescribed']} "
        f"of {section['mistakes']['fastconformer']['n']} decisive mistake sites it carries the "
        f"prescribed letter (the teacher does so at "
        f"{section['mistakes']['base']['breakdown']['prescribed']}).",
        "",
        "## Unique catches",
        "",
    ]
    if caught:
        lines += [f"- `{site_id}`" for site_id in caught]
    else:
        lines.append("None: no decisive mistake site is carried by FastConformer alone.")
    lines += [
        "",
        "## Sensitivity (committed output only; never gates)",
        "",
        "A site is flagged here only when an explicit different output is committed, so a site",
        "with nothing aligned is not flagged:",
        "",
        "| column | mistakes (sensitivity) | correct (sensitivity) |",
        "|---|---:|---:|",
    ]
    for arm in COLUMNS:
        m = section["mistakes"][arm]
        c = section["correct"][arm]
        lines.append(f"| {arm} | {m['sensitivity_flagged']}/{m['n']} | "
                     f"{c['sensitivity_flagged']}/{c['n']} |")
    lines += [
        "",
        "## Site mapping",
        "",
        f"{record['mapping']['located']} of {record['mapping']['total']} Ship A sites and "
        f"{record['mapping']['edits']['located']} of {record['mapping']['edits']['total']} edits "
        f"were located from text alone, before any decode. The gate is 95% of every site and "
        f"{record['mapping']['gates']['scored']['share']:.1f}% of the sites that have a side; "
        "the mapping never",
        "reads a decode; an unlocated site is excluded, never counted as a miss.",
        f"The {len(record['mapping']['unlocated'])} sites that did not locate are listed in",
        f"`report.json`. {record['mapping']['undecided']} sites have no human verdict "
        f"({record['mapping']['located_undecided']} of them located) and are neither scored nor",
        "in either denominator, so the mapping count and the scored count differ by exactly",
        "those two groups.",
        "",
        "## qdat",
        "",
        f"{record['qdat']['disagreeing_words']} of {record['qdat']['words']} reference words "
        f"disagree with the Imlaei text, "
        f"{record['qdat']['rates']['fastconformer']['point']:.2f} per 1,000 words (consonant "
        "and shaddah; harakat are not read). Its own column only: qdat's",
        "orthography is not compared with the harness numbers.",
        "",
        "## What the intervals cannot carry",
        "",
        f"The catch sets are small: {section['mistakes']['fastconformer']['n']} located mistake "
        f"sites and {record['edits']['edit']['n']} edits. Their intervals cannot",
        "carry a fine difference between detectors, and the correct side is a verified sample",
        f"enriched for hard sites (the base teacher's own committed text is flagged at "
        f"{section['correct']['base']['sensitivity_flagged']} of "
        f"{section['correct']['base']['n']}), not a population",
        "rate. The verdict is a screen: STOP refuses this model, GO says only worth trying, and",
        "PRIOR says the correct side passed but the catch side or the unique catches did not.",
        "",
        "## Provenance and contamination",
        "",
        "- The model is CC-BY-4.0 (attribution only); the `NOTICE` beside the decodes carries the",
        "  identity, and no audio or weights are committed.",
        "- Ship A and the edits come from Tadabur. EveryAyah's professional reciters may also be",
        "  in Tadabur; the overlap by reciter is unknown from the committed metadata, so it is",
        "  disclosed as unknown rather than resolved.",
        "- Ship A holds consonant-pair and shaddah sites only. This screen cannot certify haraka.",
        "",
    ]
    return "\n".join(lines)


def _persist(record: Mapping, report_path: Path, doc_path: Path) -> None:
    Path(report_path).write_text(json.dumps(record, ensure_ascii=False, indent=1) + "\n",
                                 encoding="utf-8")
    Path(doc_path).write_text(report_doc(record), encoding="utf-8")


def write(readout: Mapping | None = None, *, report_path: Path | None = None,
          doc_path: Path | None = None) -> dict:
    """Write ``report.json`` and the generated doc, a gate's refusal included.

    A refused mapping gate writes its STOP record (with every unlocated item) before the
    refusal propagates: the refusal is the artifact, not a traceback above a stale report.
    """
    from tadabur.fastconformer_screen.gates import ScreenStop

    report_path = REPORT_PATH if report_path is None else report_path
    doc_path = DOC_PATH if doc_path is None else doc_path
    try:
        record = dict(readout or site_report())
    except ScreenStop as stop:
        context = ({"mapping": mapping_record(located_ship_a_sites(), located_edits())}
                   if stop.gate == "mapping gate" else {})
        _persist(refusal_record(stop.gate or "gate", str(stop), **context),
                 report_path, doc_path)
        raise
    _persist(record, report_path, doc_path)
    return record


def main() -> None:
    """Write the report and the doc; a gate's refusal exits unsuccessfully after writing."""
    from tadabur.fastconformer_screen.gates import ScreenStop

    try:
        record = write()
    except ScreenStop as stop:
        print(json.dumps({"stopped": str(stop)}), flush=True)
        raise SystemExit(3)
    if record.get("refusal"):
        raise SystemExit(3)


if __name__ == "__main__":
    main()

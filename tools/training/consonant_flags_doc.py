"""Render ``docs/consonant-false-flags.md`` from ``consonant_flags/report.json`` (#155).

Every number is read from the report; the prose states only what the classification's
construction fixes (:mod:`training.consonant_flags`). Nothing here sets a threshold.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from training.consonant_flags import CLASSES, GRADING, MODEL, POPULATIONS, REFERENCE, UNCLEAR

#: Rows of each ranking shown here; the report holds every row.
TOP = 15


def _rate(row: Mapping) -> str:
    if row.get("per_1000") is None:
        return "–"
    bounds = f" [{row['lower']:.2f}, {row['upper']:.2f}]" if row.get("lower") is not None else ""
    return f"{row['per_1000']:.2f}{bounds}"


def _views(report: Mapping, population: str) -> list[Mapping]:
    return [v for v in report["populations"] if v["population"] == population]


def _class_table(report: Mapping) -> list[str]:
    out = ["| population | model | scoring | graded words | " + " | ".join(
        f"{k}: words (per 1,000)" for k in CLASSES) + " |", "|---|---|---|---|" + "---|" * len(CLASSES)]
    for v in report["populations"]:
        cells = " | ".join(f"{v['classes'][k]['words']} ({v['classes'][k]['per_1000'] or 0:.2f})" for k in CLASSES)
        out.append(f"| {v['population']} | {v['model']} | {v['scoring']} | {v['graded_words']} | {cells} |")
    return out


def _ranking(rows: Sequence[Mapping], key: str, interval: bool) -> list[str]:
    head = "per 1,000 [interval]" if interval else "per 1,000"
    out = [f"| {key} | cause | words | {head} |", "|---|---|---|---|"]
    out += [f"| {r[key]} | {r['cause']} | {r['words']} | {_rate(r)} |" for r in rows[:TOP]]
    if len(rows) > TOP:
        out.append(f"| … {len(rows) - TOP} more in the report | | | |")
    return out if rows else ["No model-class flag."]


def _not_model(rows: Sequence[Mapping]) -> list[str]:
    if not rows:
        return ["None: every located flag is at a letter Hafs pronounces there, and the decode does not hold it."]
    out = ["| class | responsible | substitution | rule | words | per 1,000 |", "|---|---|---|---|---|---|"]
    out += [f"| {r['class']} | {r['responsible']} | {r['substitution']} | {r['rule']} | {r['words']} | "
            f"{r['per_1000']:.2f} |" for r in rows]
    return out


def _population(report: Mapping, population: str) -> list[str]:
    views = _views(report, population)
    out = [f"## {population}", ""]
    for v in views:
        interval = v["interval"]
        out += [f"### {v['model']}, {v['scoring']}", "",
                f"{v['graded_words']} graded words"
                + (f" in {v['clusters']} clusters" if interval else "") + "; flags carrying a consonant "
                "error, by class (words): " + ", ".join(f"{k} {v['classes'][k]['words']}" for k in CLASSES) + ".",
                "", "**Model class, by substitution** (a word once per substitution):", "",
                *_ranking(v["model_by_substitution"], "substitution", interval), "",
                "**Model class, by expected letter:**", "",
                *_ranking(v["model_by_expected"], "expected", interval), "",
                "**Not the model (reference, tajweed or grading), with the responsible component:**", "",
                *_not_model(v["not_model"]), ""]
        if v["unclear"]:
            out += ["**Unclear** (for the owner):", "",
                    *(f"- `{word}` {sub}: {rule}" for word, sub, rule in v["unclear"]), ""]
    return out


def _owner_checks(report: Mapping) -> list[str]:
    out = ["## Owner-check list", "",
           f"Per population and model, both scorings' flags together (a word once), the five most frequent "
           f"substitutions of each class with up to ten of their words, in Quran order. The bracket marks "
           f"the flagged letter in Muraja's phonemes; `|` the word boundaries (`#` an ayah's edge).", ""]
    for entry in report["owner_checks"]:
        out += [f"### {entry['population']}, {entry['model']}", ""]
        for klass in CLASSES:
            lists = entry["checks"][klass]
            if not lists:
                continue
            out.append(f"**{klass}**")
            out.append("")
            for item in lists:
                words = "; ".join(f"`{w['word']}` {w['printed']} ({w['context']})" for w in item["words"])
                out.append(f"- {item['substitution']}: {words}")
            out.append("")
    return out


def _phonetizer_check(check: Mapping) -> list[str]:
    slips = check["slips"]
    found = (f"{len(slips)} positions where the phonetizer kept a letter the mushaf marks as not "
             "pronounced so: " + "; ".join(f"{x['ayah']} ({x['reason']})" for x in slips) + ".") if slips else (
        "none: wherever the mushaf marks a letter as not pronounced as itself, the phonetizer does "
        "not write it as itself.")
    return ["## The phonetizer against the mushaf, every ayah", "",
            f"Step 1's guard run over the whole Quran, not only the flagged words: {check['ayat_checked']} "
            f"ayat checked; {len(check['not_checked'])} not checked (the phonetizer does not reproduce "
            f"Muraja's reference there: {', '.join(check['not_checked']) or 'none'}). Slips found: {found}",
            ""]


#: The substitutions #155 suspected of being tajweed rather than mishearing.
SUSPECTED = ("ن→ل", "ل→ر", "م→ب", "ا→∅")


def _suspected(report: Mapping) -> list[str]:
    """The issue's suspected substitutions, per class, in every population and arm."""
    counts: dict[tuple, dict[str, int]] = {}
    for r in report["records"]:
        if r["substitution"] in SUSPECTED:
            row = counts.setdefault((r["population"], r["model"], r["scoring"], r["substitution"]), dict.fromkeys(CLASSES, 0))
            row[r["klass"]] += 1
    out = ["| population | model | scoring | substitution | " + " | ".join(CLASSES) + " |",
           "|---|---|---|---|" + "---|" * len(CLASSES)]
    out += [f"| {p} | {m} | {s} | {sub} | " + " | ".join(str(row[k]) for k in CLASSES) + " |"
            for (p, m, s, sub), row in sorted(counts.items(), key=lambda kv: (
                POPULATIONS.index(kv[0][0]), kv[0][1], kv[0][2], SUSPECTED.index(kv[0][3])))]
    return out


def _responsible(report: Mapping) -> str:
    """Every non-model flag of every population, by class and responsible component."""
    tally: dict[tuple, int] = {}
    for r in report["records"]:
        if r["klass"] != MODEL:
            key = (r["klass"], r["responsible"] or "none named")
            tally[key] = tally.get(key, 0) + 1
    return "; ".join(f"{k} ({who}) {n}" for (k, who), n in sorted(tally.items())) or "none"


def _headline(report: Mapping) -> list[str]:
    out = [f"- **Every population, every arm and scoring, the flags that are not the model's:** "
           f"{_responsible(report)} (consonant errors). The phonetizer is responsible for none: its "
           "letters agree with the mushaf's marks in every ayah it reproduces (below)."]
    for population in POPULATIONS:
        for v in _views(report, population):
            c = v["classes"]
            total = sum(c[k]["flags"] for k in CLASSES)
            top = ", ".join(f"{r['substitution']} {r['words']}" for r in v["model_by_substitution"][:5])
            out.append(f"- **{population}, {v['model']}, {v['scoring']}:** {total} consonant errors behind "
                       f"false flags; model {c[MODEL]['flags']}, grading {c[GRADING]['flags']}, reference "
                       f"{c[REFERENCE]['flags']}, unclear {c[UNCLEAR]['flags']}. Top model-class substitutions "
                       f"(words): {top or 'none'}.")
    return out


def render(report: Mapping) -> str:
    out = [
        "# Consonant false flags: the model's, or the reference's, tajweed's or grading's (#155)",
        "",
        "Generated by `python -m training.consonant_flags build` and `doc` (from `tools/`) from "
        "`tools/consonant_flags/report.json`; do not edit by hand. Part of #77's option B (2026-10-15): "
        f"before training is planned, every consonant false flag is classified by whose error it is. "
        f"Muraja pin `{report['muraja_commit'][:7]}`. Nothing here sets a threshold or fixes the "
        "phonetizer, Muraja or `quran.db`.",
        "",
        "## Read this first",
        "",
        *_headline(report),
        "",
        "## How a flag is classified",
        "",
        "Each consonant error the app's kept grade records for a false flag (another consonant, a "
        "target or soft pair, a collapsed geminate) is placed on Muraja's reference for its ayah "
        "(`quran.db`: the phonetizer over the whole ayah, wasl inside, waqf at its end) at the group "
        "the word scorer numbers, and read in order:",
        "",
        "1. **reference, the phonetizer:** the mushaf's marks say the letter is not pronounced as "
        "itself there (a silent letter, a hamzat al-wasl inside the ayah, an unmarked letter before "
        "a shaddah, an unmarked noon, an unmarked meem before baa, a tanween marked as not izhar or for "
        "iqlab) and the phonetizer kept it;",
        "2. **reference, Muraja's word scorer:** what was recited is known (QUD, Tadabur: the item's "
        "realized reference, its last word in waqf) and does not hold the letter there;",
        "3. **unclear:** what was recited is not known (device sessions) and the letter drops if the "
        "reader paused on the word, started at it, or joined it across an ayah boundary;",
        "4. otherwise Hafs pronounces the letter. **Grading** (Muraja's word scorer) where the "
        "committed decode holds it there and the engine flagged it anyway; **model** where the decode "
        "has another letter or nothing there. The owner's sessions have no decode in the repo: the "
        "owner's verdict stands in where the phone showed the word, else the kept grade alone.",
        "",
        "A flag whose letter is not in the word, or whose candidate letters (no group kept) are "
        "classified differently, is **unclear**. Idgham, iqlab, lam shamsiyyah and the madd and "
        "hamzat al-wasl dropped in wasl are already applied in Muraja's reference (`مِّن لَّدُنۡهُ` is "
        "`ممممِ للَدُنهُ`, `بَل رَّفَعَهُ` is `بَ ررَفَعَهُ`), so a flag on them names the letter the "
        "reference holds, not the one the spelling shows.",
        "",
        *_phonetizer_check(report["phonetizer_check"]),
        "## The substitutions #155 suspected",
        "",
        "Consonant errors per class. Muraja's reference already applies the assimilation these "
        "were suspected of (above), so each names a letter the reference itself holds.",
        "",
        *_suspected(report),
        "",
        "## Classes per population",
        "",
        *_class_table(report),
        "",
    ]
    for population in POPULATIONS:
        out += _population(report, population)
    out += _owner_checks(report)
    out += [
        "## What this can and cannot say",
        "",
        "- **Can:** for each consonant false flag in these populations, whether Hafs pronounces the "
        "expected letter there by the phonetizer's rules and the mushaf's marks, and whether the "
        "committed decode holds it.",
        "- **Cannot:** whether a professional reciter actually pronounced every letter (the QUD "
        "recitation is presumed correct, nobody listened); the check windows the engine graded "
        "differ from the committed transcript, so *grading* is read from the transcript the app "
        "commits, not from the window that set the kept grade, and by the truth-site scorer's carrier "
        "rule, so a decode that holds the letter next to an inserted one, or one half of a geminate "
        "the scorer split, can read as holding it (the grading list is for the owner to check); "
        "device sessions' pauses are unknown.",
        "- QUD's intervals are clustered by mushaf and surah (two reciters: they describe these "
        "recordings, not reciters); the device logs' by log; the owner's sessions, the word sample "
        "and the held-out verified sites (selected at target-pair swaps, so their rates are not "
        "population rates) carry counts and points only.",
        "",
    ]
    return "\n".join(out)

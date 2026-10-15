"""Render ``docs/nonpro-pilot.md`` from the pilot's report and the overlap sample (#156).

Every number is read from ``tadabur/nonpro_eval/report.json`` and ``overlap.json``; the prose
states only what their construction fixes. Nothing here recommends a threshold, a margin or a
listening budget.
"""

from __future__ import annotations


from training.carrier_windowing_doc import rate_cell
from training.direction_table import FAMILIES, GEMINATE, SWAP

#: The guard's floor #144 sets the per-pair counts against (acceptance rules, n >= 189).
FLOOR = 189
SOURCES = ("tarteel_v1", "retasy", "qdat")
NAMES = {"tarteel_v1": "Tarteel v1", "retasy": "RetaSy", "qdat": "qdat", "qdat_bench": "qdat_bench"}
MODELS = ("base", "h448")
SCORINGS = ("shipped_default", "allowances_off")
ARMS = ("base/spans", "base/stream_b0", "base/checks", "h448/spans", "h448/stream_b0",
        "h448/checks")
#: Substitutions listed per source, model and scoring: those carried by at least this many words.
MIN_WORDS = 3


def _n(x) -> str:
    return "–" if x is None else f"{x:.1f}"


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def _classes(counts: dict) -> str:
    return (f"{counts.get('agreed', 0)} / {counts.get('disagreed', 0)} / "
            f"{counts.get('abstained', 0)}")


def _sources(report: dict) -> list[str]:
    out = ["## The corpora", "",
           "| source | dataset (revision) | licence | per-clip outputs committed | rows | "
           "selected | left out |", "|---|---|---|---|---|---|---|"]
    for name, d in report["datasets"].items():
        s = report.get("selection", {}).get(name)
        left = ("; ".join(f"{reason} {n}" for reason, n in s["left_out"].items()) or "none") \
            if s else "(its clips are qdat's)"
        committed = ("yes" if d["per_clip_outputs_committed"]
                     else "no: aggregates and clip ids only")
        out.append(f"| {NAMES[name]} | `{d['repo']}` (`{d['revision'][:8]}`) | {d['licence']} | "
                   f"{committed} | {s['rows'] if s else 159} | {s['selected'] if s else 159} | "
                   f"{left} |")
    out += ["", "| source | clips decoded | phonetizer cannot handle | presumed correct | hours "
            "| ayat | surahs | reciters | words |", "|---|---|---|---|---|---|---|---|---|"]
    for name in SOURCES:
        i = report["sources"][name]["items"]
        out.append(f"| {NAMES[name]} | {i['clips_decoded']} | {i['phonetizer_unsupported']} | "
                   f"{i['presumed_correct']} | {i['hours']} | {i['ayat']} | {i['surahs']} | "
                   f"{i['reciters'] or 'unknown'} | {i['words']} |")
    retasy = report["sources"]["retasy"]["items"]["labels"]
    p = report["passage"]
    out += ["",
            "Each clip is one item: the whole clip, read against the phonetizer's realized "
            "reference for the canonical words it recites, last word in waqf. Tarteel v1's text is "
            "undiacritised, so its clips are placed by their `sura`/`ayah` columns (the whole "
            "ayah); RetaSy's by matching its text to a run of the surah's words (its prayers and "
            "adhan are no Quran and are left out). Tarteel v1 kept only clips two ASR models "
            "transcribe with CER under 0.3, so it leans towards easier, cleaner clips. RetaSy's "
            f"Hafs clips read are labelled `correct` ({retasy.get('correct', 0)}) or `in_correct` "
            f"({retasy.get('in_correct', 0)}); only the `correct` ones are presumed correct, the "
            "others are read only beside their label.",
            "",
            f"**Which passage qdat recites.** Its card quotes 2:32's wording, qdat_bench's 5:109. "
            f"Of {p['clips']} clips, the base teacher's whole-clip decode is nearer 5:109's words "
            f"8-15 for {p['nearer'].get('5:109 words 8-15', 0)} and nearer 2:32 for "
            f"{p['nearer'].get('2:32', 0)} (median normalized edit distance "
            f"{p['median_distance']['5:109 words 8-15']} against "
            f"{p['median_distance']['2:32']}): the passage is 5:109's "
            "`قَالُوا۟ لَا عِلْمَ لَنَآ إِنَّكَ أَنتَ عَلَّٰمُ ٱلْغُيُوبِ`. qdat_bench's 159 clips are "
            "byte-identical to the qdat clips of the same id, so they are read from qdat's decodes.",
            ""]
    return out


def _overlap(overlap: dict) -> list[str]:
    out = ["## Overlap (a cheap check)", "",
           "These are app-user and amateur recordings, presumed independent of Tadabur, Muaalem "
           "and EveryAyah. The check is a sample (`tadabur.nonpro_overlap`): a few parquet row "
           "groups per source drawn by a salted hash, up to ten decoded clips in each, each "
           f"fingerprinted against the same ayah in every EveryAyah reciter folder "
           f"({overlap['everyayah_folders']}), at #144's line (BER ≤ "
           f"{overlap['fingerprint_line']}, fixed from #144's controls, not retuned).", "",
           "| source | clips sampled | recordings compared | lowest BER | matches |",
           "|---|---|---|---|---|"]
    for name in SOURCES:
        o = overlap["datasets"][name]
        out.append(f"| {NAMES[name]} | {o['sampled']} | {o['comparisons']} | {o['lowest_ber']} | "
                   f"{o['matches']} |")
    out += ["",
            "**Zero matches means no overlap detected in the sample, not proven absence.** "
            "Tadabur is not read: its rows of one ayah are spread over 385 shards (#144's screen "
            "read every shard, 8 h), and its 618 reciters, like Muaalem's 27 moshafs, are named "
            "professionals; EveryAyah stands in for a published professional recording. Every "
            "decoded clip, and every qdat_bench clip, is registered in the exposure registry as "
            "the evaluation-only use `nonpro.evaluation_only` "
            "(`tools/tadabur/exposure/nonpro.evaluation_only.clips.jsonl`), which "
            "`check_disjoint` refuses to certify anything with.", ""]
    return out


def _directions(report: dict) -> list[str]:
    out = ["## Sites per direction, against the mushaf (no listening)", "",
           "#144's rule reads two decodes: the base teacher on the whole clip and `h448` "
           "streamed at b = 0. **Agreed**: both commit the mushaf's letter (hold the geminate): "
           "a truth site with `assumes_competent_reciter`; **disagreed**: either commits "
           "something else: a `pending` site and a worklist row, not queued; **abstained**: "
           "neither disagrees, one commits nothing (counted only). A carrier in two pairs counts "
           f"in both directions. Agreed counts at or above the n ≥ {FLOOR} floor are marked ✓. "
           "Each cell is agreed / disagreed / abstained (disagreed: base committed another, "
           "`h448` committed another).", "",
           "| direction | " + " | ".join(NAMES[n] for n in SOURCES) + " |",
           "|---|" + "---|" * len(SOURCES)]
    order = [d for family in FAMILIES.values() for d in family]
    for d in order:
        cells = []
        for name in SOURCES:
            c = report["sources"][name]["directions"].get(d)
            if c is None:
                cells.append("–")
                continue
            mark = " ✓" if c.get("agreed", 0) >= FLOOR else ""
            cells.append(f"{c.get('agreed', 0)}{mark} / {c.get('disagreed', 0)} / "
                         f"{c.get('abstained', 0)} ({c.get('base/spans committed another', 0)}, "
                         f"{c.get('h448/stream_b0 committed another', 0)})")
        out.append(f"| {'in-word geminate' if d == GEMINATE else d} | " + " | ".join(cells) + " |")
    out += ["", "Carriers per class (each carrier once): " + "; ".join(
        f"{NAMES[n]} " + ", ".join(f"{k} {v}" for k, v in
                                   sorted(report["sources"][n]["classes"].items()))
        for n in SOURCES) + ". Truth sites: "
        "`tools/tadabur/nonpro_eval/<source>/truth_sites.jsonl.gz` (Tarteel v1, qdat; RetaSy's "
        "are not committed).", ""]
    return out


def _consonants(report: dict) -> list[str]:
    out = ["## Every consonant (option B)", "",
           "The same rule at the first letter of every run of every consonant, whatever its "
           "pair. A truth site names a mark under test and \"this consonant\" is none, so these "
           "carriers are counted and their disagreements listed "
           "(`<source>/consonant_worklist.jsonl`), never written as sites. Each cell is agreed "
           "/ disagreed / abstained.", "",
           "| consonant | " + " | ".join(NAMES[n] for n in SOURCES) + " |",
           "|---|" + "---|" * len(SOURCES)]
    letters = sorted({c for n in SOURCES for c in report["sources"][n]["consonants"]["letters"]})
    for letter in letters:
        cells = []
        for name in SOURCES:
            c = report["sources"][name]["consonants"]["letters"].get(letter)
            cells.append("–" if c is None else _classes(c))
        out.append(f"| {letter} | " + " | ".join(cells) + " |")
    out += ["| all | " + " | ".join(_classes(report["sources"][n]["consonants"]["total"])
                                   for n in SOURCES) + " |", "",
        "What each rule arm committed instead at the disagreements (expected → committed, the ten "
        "most frequent):", ""]
    for name in SOURCES:
        for a, rows in report["sources"][name]["consonants"]["committed_instead"].items():
            out.append(f"- {NAMES[name]}, {a}: " + (", ".join(
                f"{r['substitution']} {r['carriers']}" for r in rows[:10]) or "none") + ".")
    return out + [""]


def _rates(report: dict) -> list[str]:
    out = ["## Swap rates per model and protocol", "",
           "Per direction, the share of carriers where each arm committed the pair's partner, "
           "or a single consonant at a geminate (`swap`), against the mushaf. Intervals are "
           "clustered by reciter (RetaSy, qdat) or, where no reciter is named (Tarteel v1), by "
           "surah.", ""]
    for name in SOURCES:
        rows = report["sources"][name]["rates"]["rows"]
        out += [f"### {NAMES[name]}", "", "| direction | sites | " + " | ".join(ARMS) + " |",
                "|---|---|" + "---|" * len(ARMS)]
        for row in rows:
            if not row.get("arms"):
                continue
            label = row["direction"] if row["direction"] != "all" else row["family"] + " (all)"
            out.append(f"| {label} | {row['sites']} | " + " | ".join(
                rate_cell(row["arms"][a][SWAP], row["counts"][a][SWAP]) for a in ARMS) + " |")
        out.append("")
    return out


def _app(report: dict) -> list[str]:
    app = report["app"]
    out = ["## The app outcome on presumed-correct non-professional recitation", "",
           "Every word of every presumed-correct clip, graded by Muraja's engine (protocol 3, "
           "Muraja at `99c326f`) on the clip's recorded check sequence, per model; each session "
           "starts on the clip's first canonical word. **This is not truth:** nobody has heard "
           "these clips, so a flagged word is a flag on presumed-correct recitation, and some "
           "flags are the reciter's real slips.", "",
           "| source | model | scoring | words graded | flagged | per 1,000 | " +
           " | ".join(["tashkeel", "soft_pair", "target_pair", "other_consonant", "shaddah",
                       "skipped", "unrecorded"]) + " |", "|---|---|---|---|---|---|" + "---|" * 7]
    for name in SOURCES:
        for model in MODELS:
            for scoring in SCORINGS:
                v = app["outcome"][name][model][scoring]
                out.append(f"| {NAMES[name]} | {model} | {scoring} | {v['graded']} | "
                           f"{v.get('flagged', 0)} | {_n(v['per_1000'])} | " +
                           " | ".join(str(c) for c in v["causes"].values()) + " |")
    out += ["", "### Beside the other populations", "",
            "Flags per 1,000 graded words (flags / words). QUD: two professional mushafs (#144). "
            "Device: Muraja's own sessions with the teacher (#151), explained words left out. "
            "Owner: the owner's three `h448` sessions (#153), the owner's mistakes left out. Word "
            "sample: the Tadabur word sample's correctly recited words (#90), unweighted, with "
            "the weighted rate after it.", "",
            "| model | scoring | " + " | ".join(NAMES[n] for n in SOURCES) +
            " | QUD (#144) | device (#151) | owner (#153) | word sample (#90) |",
            "|---|---|" + "---|" * (len(SOURCES) + 4)]

    def cell(r) -> str:
        return "–" if r is None else f"{_n(r['per_1000'])} ({r['flags']} / {r['words']})"

    for model in MODELS:
        for scoring in SCORINGS:
            c = app["comparison"][model][scoring]
            sample = c["word_sample_90"]
            out.append(f"| {model} | {scoring} | " + " | ".join(cell(c[n]) for n in SOURCES) +
                       f" | {cell(c['qud_144'])} | {cell(c['device_151_teacher'])} | "
                       f"{cell(c['owner_153_h448'])} | {cell(sample)}, weighted "
                       f"{_n(sample['weighted_per_1000'])} |")
    out += ["", "**Consonant flags by substitution** (expected → heard; a deletion is `→∅`; a "
            f"word counts once per substitution; `word_flags.substitution_tally`), those carried "
            f"by at least {MIN_WORDS} words:", ""]
    for name in SOURCES:
        for model in MODELS:
            for scoring in SCORINGS:
                rows = [r for r in app["outcome"][name][model][scoring]["substitutions"]
                        if r["words"] >= MIN_WORDS]
                out.append(f"- {NAMES[name]}, {model}, {scoring}: " + (", ".join(
                    f"{r['substitution']} {r['words']} ({r['cause']})" for r in rows) or "none")
                    + ".")
    out += ["", "By expected letter (every consonant flag, top ten per source, model and the "
            "shipped default):", ""]
    for name in SOURCES:
        for model in MODELS:
            rows = app["outcome"][name][model]["shipped_default"]["substitutions_by_expected"]
            out.append(f"- {NAMES[name]}, {model}: " + (", ".join(
                f"{r['expected']} {r['words']} ({r['cause']})" for r in rows[:10]) or "none") + ".")
    return out + [""]


def _qdat(report: dict) -> list[str]:
    q = report["app"]["qdat"]
    out = ["## qdat: one passage, many amateur voices", "",
           "Every reciter recorded 5:109's words 8-15 about ten times, so each word of the "
           "passage is graded on every clip. Per Muraja word, the shipped default:", "",
           "| word | text | " + " | ".join(f"{m} flagged / graded (per 1,000)" for m in MODELS)
           + " | " + " | ".join(f"{m} substitutions" for m in MODELS) + " |",
           "|---|---|" + "---|" * (2 * len(MODELS))]
    per_word = {m: q["per_word"][m]["shipped_default"] for m in MODELS}
    for k, w in enumerate(per_word["base"]):
        cells = [f"{per_word[m][k]['flagged']} / {per_word[m][k]['graded']} "
                 f"({_n(per_word[m][k]['per_1000'])})" for m in MODELS]
        subs = [", ".join(f"{r['substitution']} {r['words']}"
                          for r in per_word[m][k]["substitutions"][:5]) or "–" for m in MODELS]
        out.append(f"| {w['word']} | {w['text']} | " + " | ".join(cells) + " | " +
                   " | ".join(subs) + " |")
    out += ["", "**Per reciter** (shipped default): how many of the reciters ever get a given "
            "flag, by cause and by substitution (the ten most widespread):", ""]
    for model in MODELS:
        r = q["per_reciter"][model]["shipped_default"]
        out.append(f"- {model}: {r['reciters_flagged']} of {r['reciters']} reciters have a "
                   "flagged word. By cause: " + ", ".join(
                       f"{x['cause']} {x['reciters']} reciters ({x['words']} words)"
                       for x in r["by_cause"]) + ". By substitution: " + ", ".join(
                       f"{x['substitution']} {x['reciters']} ({x['words']})"
                       for x in r["by_substitution"][:10]) + ".")
    out += ["", "**Split by qdat's tajweed labels** (shipped default, flagged / graded, per "
            "1,000). The labels mark tajweed lapses (madd, the tight noon, concealment, and "
            "`target`, which its card's author did not explain either), **not consonant "
            "errors**, so they are no consonant truth:", "",
            "| label | value | " + " | ".join(MODELS) + " |", "|---|---|" + "---|" * len(MODELS)]
    labels = q["per_tajweed_label"]["base"]["shipped_default"]
    for label, values in labels.items():
        for value in values:
            out.append(f"| {label} | {value} | " + " | ".join(
                "{flagged} / {graded} ({rate})".format(
                    rate=_n(q["per_tajweed_label"][m]["shipped_default"][label][value]["per_1000"]),
                    **q["per_tajweed_label"][m]["shipped_default"][label][value])
                for m in MODELS) + " |")
    return out + [""]


def _bench(report: dict) -> list[str]:
    letters, app = report["bench"]["letters"], report["app"]["bench"]
    out = ["## qdat_bench: the experts' transcript", "",
           f"Each of the bench's {letters['clips']} clips has an expert-annotated "
           "`phonetic_transcript` (quran-transcript's script, madd lengths and ghunnah "
           "included). Each arm's decode is read against it letter by letter: a consonant at "
           "the first letter of each run, a haraka at each consonant a fatha, damma or kasra "
           "follows, a shaddah at each geminate. Agreement is agree / all; committed agreement "
           "leaves out the slots the decode left empty. Aggregates only (its licence is "
           "unstated).", "",
           "| arm | consonant | haraka | shaddah |", "|---|---|---|---|"]

    def cell(c: dict) -> str:
        return (f"{_pct(c['agreement'])} (committed {_pct(c['agreement_committed'])}; "
                f"{c['agree']} / {c['other']} / {c['none']})")

    for a, kinds in letters["arms"].items():
        out.append(f"| {a} | " + " | ".join(cell(kinds[k]) for k in ("consonant", "haraka",
                                                                      "shaddah")) + " |")
    t = letters["transcript_vs_mushaf"]
    out += ["", "Cells: agreement (committed agreement; agree / other / empty).", "",
            "**The transcript against the mushaf** (where the experts heard the reciter leave "
            "it): " + "; ".join(f"{k} {cell(t[k])}" for k in ("consonant", "haraka", "shaddah"))
            + ". Clips that leave the mushaf at least once: " + ", ".join(
                f"{k} {v}" for k, v in t["clips_leaving_the_mushaf"].items()) + ".", "",
            "**The app outcome on the bench's clips**, each graded word placed by whether the "
            "experts' transcript leaves the mushaf in it:", "",
            "| model | scoring | words graded | flagged (per 1,000) | words where the experts "
            "heard a departure | flagged there | flagged where they heard the mushaf (of "
            "graded) |", "|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for scoring in SCORINGS:
            v = app[model][scoring]
            out.append(f"| {model} | {scoring} | {v['graded']} | {v['flagged']} "
                       f"({_n(v['per_1000'])}) | {v['words_where_experts_heard_a_departure']} | "
                       f"{v['flagged_where_experts_heard_a_departure']} | "
                       f"{v['flagged_where_experts_heard_the_mushaf']} (of "
                       f"{v['graded_where_experts_heard_the_mushaf']}) |")
    return out + [""]


def _retasy(report: dict) -> list[str]:
    r = report["app"]["retasy_labels"]
    out = ["## RetaSy: the app beside its human label", "",
           "RetaSy labels whole clips `correct` or `in_correct` \"with the diacritics, "
           "regardless of tajweed\" (crowd majority, or an expert for its golden clips). A "
           "sanity check, not a measurement: crowd accuracy is 0.77 on its card, and a clip "
           "label says nothing about which word. Aggregates only (its licence is unknown).", "",
           "| model | scoring | label | clips | clips with a flagged word | words graded | "
           "flagged | per 1,000 |", "|---|---|---|---|---|---|---|---|"]
    for model in MODELS:
        for scoring in SCORINGS:
            for label, v in r[model][scoring].items():
                out.append(f"| {model} | {scoring} | {label} | {v['clips']} | "
                           f"{v['clips_flagged']} | {v['graded']} | {v['flagged']} | "
                           f"{_n(v['per_1000'])} |")
    return out + [""]


def _listening(report: dict) -> list[str]:
    out = ["## The owner's listening, from the measured disagreements", "",
           "No corpus here times its words, so each disagreed carrier's excerpt is its whole "
           "clip, played twice plus 4 s to answer (#87's model). A budget of *k* draws at most "
           "*k* disagreed carriers per stratum (a carrier letter, or the in-word geminates; a "
           "consonant for the every-consonant worklist), at the stratum's mean clip time. No "
           "sample is drawn here.", "",
           "| source | worklist | disagreed carriers | census | k = 20 | k = 50 |",
           "|---|---|---|---|---|---|"]
    for name in SOURCES:
        for kind, label in (("pairs_and_geminates", "target pairs and geminates"),
                            ("consonants", "every consonant")):
            e = report["sources"][name]["listening"][kind]["pooled"]
            out.append(f"| {NAMES[name]} | {label} | {e['disagreed_carriers']} | "
                       f"{e['census_minutes']} min | {e['k20_minutes']} min | "
                       f"{e['k50_minutes']} min |")
    return out + [""]


def _compute(report: dict) -> list[str]:
    c = report["compute"]
    return ["## Compute and disk", "",
            f"- GPU: {c['gpu_hours']} h of wall time holding the GPU lock ({c['parallel_runs']} "
            f"shard runs side by side on one {c['gpu']}), for {c['audio_hours']} h of clip audio "
            "decoded by both models in all three protocols.",
            f"- Peak scratch disk on cuda-dev: {c['peak_scratch_gb']} GB, sampled each minute from "
            f"{c['disk_sampled_from']} (the shards being read, each deleted once read, and the "
            "decodes); no audio was kept.", ""]


def render(report: dict, overlap: dict) -> str:
    lines = [
        "# Non-professional eval pilot: app users and amateurs, measured without listening (#156)",
        "",
        "Generated by `python -m training.nonpro_eval doc` from "
        "`tools/tadabur/nonpro_eval/report.json` and `overlap.json`. Evaluation only, never a "
        "training signal; nothing here sets a threshold. Three corpora of **non-professional** "
        "recitation, read with QUD's pipeline (`training.external_corpus`, #144): Tarteel v1 app "
        "users, RetaSy's non-Arab volunteers on phones, and qdat's amateur reciters, with "
        "qdat_bench's expert transcripts of 159 qdat clips.",
        "",
        *_sources(report),
        *_overlap(overlap),
        *_directions(report),
        *_consonants(report),
        *_app(report),
        *_qdat(report),
        *_bench(report),
        *_retasy(report),
        *_rates(report),
        *_listening(report),
        *_compute(report),
        "## What this can and cannot say",
        "",
        "- **Can:** how often each model commits a letter other than the mushaf's, at every "
        "target-pair carrier, in-word geminate and consonant, on recitation by app users and "
        "amateurs neither model heard; how many words the app flags there; and, on qdat_bench, "
        "how each decode agrees with expert transcripts letter by letter.",
        "- **Cannot:** whether a disagreement or a flag is a model error or the reciter's: "
        "nobody has listened, and non-professionals do slip, so every disagreement is "
        "`pending` and every flag is a flag on presumed-correct recitation, not a measured "
        "false flag. qdat_bench's experts are the one exception, on 159 clips of one passage. "
        "Tarteel v1 keeps only clips two ASR models transcribe well, which leans towards easier "
        "clips. Its users are unnamed, so its intervals are clustered by surah.",
        "- **Overlap:** no sampled clip matched an EveryAyah recording; Tadabur was not read "
        "(see the overlap section).",
        "- Nothing here trains, sets a threshold, a margin or a floor, or touches the sealed "
        "panel.",
        "",
    ]
    return "\n".join(lines)


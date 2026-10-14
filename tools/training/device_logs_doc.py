"""Render the device-log false-flag report (``docs/device-log-false-flags.md``) from its JSON.

Presentation only: every number comes from :func:`training.device_logs.build_report`, and the doc
is regenerated with it by :mod:`training.device_logs`, never edited by hand. The one hand-written
block is :data:`H448_SHIP_EVIDENCE`, the dated history behind the model each log is assigned.
"""

from __future__ import annotations

from collections import Counter, defaultdict

from training.device_logs import (
    ALL_WORDS,
    DOCUMENTED_MISTAKE,
    GROUPS,
    H448,
    MODELS,
    OVERLAP_CAP,
    STUDENT_INFERENCE_MS,
    TEACHER,
    TEACHER_ONE_SECOND,
    UNEXPLAINED,
    position,
)
from training.word_flags import CAUSES

CAUSE_TITLES = {
    "tashkeel": "tashkeel",
    "soft_pair": "soft-pair consonant",
    "target_pair": "non-soft target pair (ذ↔ظ, س↔ش, ط↔ق)",
    "other_consonant": "another consonant",
    "shaddah": "shaddah (collapsed geminate)",
    "skipped": "skipped by the engine",
    "unrecorded": "no error recorded",
}
GROUP_TITLES = {
    H448: "`h448`",
    TEACHER: "the six-chunk teacher",
    TEACHER_ONE_SECOND: "the six-chunk teacher, 1 s-hop sessions only",
}
VIEW_TITLES = {ALL_WORDS: "every graded word", UNEXPLAINED: "explained words left out"}
EXCLUSION_TITLES = {
    "synthetic": "synthetic (reference phonemes, no model)",
    "not_a_device_session": "no device session (mined Tadabur clips transcribed offline)",
    "duplicate_session": "a duplicate session",
    "documented_mistake": "a slip into another passage, documented",
}

H448_SHIP_EVIDENCE = """\
- HifzGuide release `v1.0.5` (2026-09-21 18:56 UTC) replaced `models.zip`'s six 6-bit teacher
  chunks with `MuaalemStudent_h448_6BIT`, "published for on-device testing". Releases `v1.0.0`
  (2026-04-08) to `v1.0.4` all carry the same six-chunk `models.zip` (one sha256, `9a3590cb…`).
- Muraja `796fc28` ("load the single distilled student, not six teacher chunks") and the merge
  `fc23225` ("single distilled student model (HifzGuide v1.0.5)", 2026-09-21 19:31 −0700,
  "device-tested") put `h448` on Muraja `main`. Every build from `main` after that runs `h448`.
- The first release after it is 1.0.22 (Muraja `81c84d5`, 2026-09-27; PR #182: "the model is now
  bundled"). Release 1.0.21 (`dbcbc1c`, 2026-08-24) predates it and ran the teacher.
"""


def _inference_evidence(report: dict) -> str:
    """The last line of the ship evidence: each model's range of median inference times."""
    ranges = {}
    for model in MODELS:
        times = [e["median_inference_ms"] for e in report["logs"]
                 if e["model"] == model and e["median_inference_ms"] is not None]
        ranges[model] = (len(times), min(times), max(times))
    (n_s, lo_s, hi_s), (n_t, lo_t, hi_t) = ranges[H448], ranges[TEACHER]
    return (f"- The logs agree: the {n_s} sessions captured after 2026-09-21 ran their inferences in a\n"
            f"  median {lo_s:.0f}–{hi_s:.0f} ms, the {n_t} earlier ones that log the time in {lo_t:.0f}–{hi_t:.0f} ms\n"
            "  (the teacher runs six chunks in turn); three earlier logs also record the six teacher\n"
            "  chunks loading. The generator refuses an inventory whose model contradicts a log's\n"
            f"  inference time (the line is drawn at {STUDENT_INFERENCE_MS} ms).\n")


def _num(value, digits: int = 1) -> str:
    return "–" if value is None else f"{value:.{digits}f}"


def fmt_rate(rate: dict) -> str:
    """``flags / words · per 1,000 [interval]``; ``n/c`` where no interval is admissible."""
    if not rate["words"]:
        return "–"
    bounds = (f"[{_num(rate['lower'])}, {_num(rate['upper'])}]" if rate["method"] != "none"
              else "(n/c)")
    return f"{rate['flags']} / {rate['words']} · {_num(rate['per_1000'])} {bounds}"


def _scoring_names(report: dict) -> list[str]:
    return [s["name"] for s in report["scorings"]]


def _table(header: list[str], rows: list[list[str]]) -> list[str]:
    return ["| " + " | ".join(header) + " |", "|" + "---|" * len(header),
            *("| " + " | ".join(row) + " |" for row in rows), ""]


def _scorings_section(report: dict) -> list[str]:
    rows = [[f"`{s['name']}`", f"`{s['mode']}`",
             {None: "on (balanced)", True: "on", False: "**off**"}[s["soft_pairs_enabled"]],
             {None: "on (balanced)", True: "on", False: "**off**"}[s["shaddah_suppression"]],
             "shown" if s["tashkeel_shown"] else "hidden"] for s in report["scorings"]]
    return [
        "## The scorings",
        "",
        "Each log is replayed once per engine configuration; the Tashkeel toggle is a display",
        "filter applied to the kept grade (`muraja_policy.displayed`). `shipped_default` is the app",
        "as installed; `allowances_on` is what a reader who turned tashkeel detection off sees.",
        "",
        *_table(["scoring", "mode", "soft-pair forgiveness", "shaddah suppression", "tashkeel"], rows),
    ]


def _rates_section(report: dict) -> list[str]:
    names = _scoring_names(report)
    lines = [
        "## False flags per 1,000 words",
        "",
        "Count / graded words · flags per 1,000 [interval]. The interval resamples logs (the words",
        "of one session are not independent); with too few logs none is admissible (n/c). The",
        "second column leaves out the explained words (skipped, re-read or never recited, from the",
        "log's docs or its own decode; see *Explained words*) from both counts.",
        "",
    ]
    for group in GROUPS:
        by_scoring = report["groups"][group]
        logs = by_scoring[names[0]][ALL_WORDS]["false_flags"]["logs"]
        lines += [f"**{GROUP_TITLES[group]}** ({logs} log{'' if logs == 1 else 's'})", ""]
        lines += _table(["scoring", VIEW_TITLES[ALL_WORDS], VIEW_TITLES[UNEXPLAINED]],
                        [[f"`{n}`", fmt_rate(by_scoring[n][ALL_WORDS]["false_flags"]),
                          fmt_rate(by_scoring[n][UNEXPLAINED]["false_flags"])] for n in names])
    lines += [
        "### What the unexplained false flags are",
        "",
        "Flags per cause over the graded words, explained words left out (count · per 1,000). A",
        "word may carry several causes. Below each table: the tashkeel marks (the mark the mushaf",
        "prescribes; `dropped` = no mark, `other` = another mark) and the target pairs behind the",
        "consonant flags.",
        "",
    ]
    for group in GROUPS:
        by_scoring = {n: report["groups"][group][n][UNEXPLAINED] for n in names}
        rows = [[CAUSE_TITLES[c], *(f"{by_scoring[n]['causes'][c]['flags']} · "
                                     f"{_num(by_scoring[n]['causes'][c]['per_1000'])}" for n in names)]
                for c in CAUSES]
        details = sorted({(kind, key) for n in names for kind in ("marks", "pairs")
                          for key in by_scoring[n][kind]})
        rows += [[f"— {key}", *(str(by_scoring[n][kind].get(key, 0)) for n in names)]
                 for kind, key in details]
        lines += [f"**{GROUP_TITLES[group]}**", ""] + _table(["", *(f"`{n}`" for n in names)], rows)
    return lines


def _flag_cell(row: dict | None) -> str:
    if row is None:
        return "correct"
    detail = ", ".join(row["marks"] + row["pairs"])
    return f"{row['quality']}: {', '.join(row['causes'])}" + (f" ({detail})" if detail else "")


def _word_table(report: dict, logs: set[str], explained: bool) -> list[str]:
    names = _scoring_names(report)
    model = {entry["log"]: entry["model"] for entry in report["logs"]}
    cells: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in report["words"]:
        if row["log"] in logs and (row["explained"] is not None) == explained:
            cells[(row["log"], row["word"])][row["scoring"]] = row
    rows = [[f"`{log}`", word, model[log], *(_flag_cell(by.get(n)) for n in names)]
            for (log, word), by in sorted(cells.items(), key=lambda kv: (kv[0][0], position(kv[0][1])))]
    return _table(["log", "word", "model", *(f"`{n}`" for n in names)], rows) if rows else ["None.", ""]


def _comparison_section(report: dict) -> list[str]:
    names = _scoring_names(report)
    lines = [
        "## The device against the simulation (#90)",
        "",
        "#90's word sample: 155 correctly recited words, one per held-out Tadabur reciter, replayed",
        "on their recorded check sequences (`docs/truth-baseline.md`). Its rate is shown as the count",
        "over 155 and as the weighted rate per 1,000 with its interval. The device side is the",
        "unexplained view. The last column is how many of the simulation's words the device's rate",
        "would flag: 155 × the device's rate.",
        "",
    ]
    for model in MODELS:
        device_group = TEACHER_ONE_SECOND if model == TEACHER else H448
        device = report["groups"][device_group]
        sim = report["simulation"][model]
        lines += [f"**{GROUP_TITLES[device_group]}** against the simulation's "
                  f"`{'base' if model == TEACHER else 'h448'}/checks`", ""]
        rows = []
        for n in names:
            d = device[n][UNEXPLAINED]["false_flags"]
            s = sim[n]
            expected = s["words"] * d["per_1000"] / 1000 if d["per_1000"] is not None else None
            rows.append([f"`{n}`", fmt_rate(d),
                         f"{s['flags']} / {s['words']} · {_num(1000 * s['flags'] / s['words'])} "
                         f"(weighted {_num(s['per_1000'])} [{_num(s['lower'])}, {_num(s['upper'])}])",
                         _num(expected)])
        lines += _table(["scoring", "device", "simulation", "device rate × 155"], rows)
        cause_rows = []
        for c in CAUSES:
            cells = []
            for n in ("allowances_on", "allowances_off"):
                d = device[n][UNEXPLAINED]["causes"][c]
                s = sim[n]["causes"][c]
                expected = sim[n]["words"] * d["per_1000"] / 1000 if d["per_1000"] is not None else None
                cells += [f"{d['flags']} / {d['words']}", f"{s['flags']} / {sim[n]['words']}", _num(expected)]
            cause_rows.append([CAUSE_TITLES[c], *cells])
        lines += _table(["cause", "device, on", "simulation, on", "× 155, on",
                         "device, off", "simulation, off", "× 155, off"], cause_rows)
    return lines + _disagreements(report)


def _recurring(report: dict, cause: str, scoring: str, logs: set[str]) -> list[tuple[str, int]]:
    """The words an unexplained cause falls on under a scoring, and in how many sessions."""
    sessions: dict[str, set[str]] = defaultdict(set)
    for row in report["words"]:
        if row["log"] in logs and row["scoring"] == scoring and not row["explained"] and cause in row["causes"]:
            sessions[row["word"]].add(row["log"])
    return sorted(((w, len(s)) for w, s in sessions.items()), key=lambda ws: (-ws[1], position(ws[0])))


def _word_list(words: list[tuple[str, int]]) -> str:
    return ", ".join(f"{w} ({n} sessions)" if n > 1 else w for w, n in words) or "no word"


#: The allowance-sensitive causes, and the allowance each answers to.
ALLOWANCE_CAUSES = (("soft_pair", "soft-pair forgiveness"), ("shaddah", "shaddah suppression"),
                    ("tashkeel", "the Tashkeel toggle"))


def _disagreements(report: dict) -> list[str]:
    included = [e for e in report["logs"] if e["exclusion"] is None]
    names = {e["log"] for e in included}
    teacher = report["groups"][TEACHER]
    on, off = "allowances_on", "allowances_off"
    explained = 1 - (teacher[off][UNEXPLAINED]["false_flags"]["flags"]
                     / max(1, teacher[off][ALL_WORDS]["false_flags"]["flags"]))
    with_skips = sum(any(x["kind"] == "skip" for x in e["explained"]) for e in included)
    def against(group: str, model: str) -> str:
        device_rate = report["groups"][group][on][UNEXPLAINED]["false_flags"]["per_1000"]
        sim = report["simulation"][model][on]
        return f"{_num(device_rate)} against {_num(1000 * sim['flags'] / sim['words'])}"

    page, sessions = max(sorted(Counter(p for e in included for p in e["pages"]).items()), key=lambda kv: kv[1])
    added = [f"{CAUSE_TITLES[c]} {teacher[off][UNEXPLAINED]['causes'][c]['flags'] - teacher[on][UNEXPLAINED]['causes'][c]['flags']}"
             f" (on {_word_list(_recurring(report, c, off, names))})" for c, _ in ALLOWANCE_CAUSES]
    expected = {}
    for model in MODELS:
        group = report["groups"][TEACHER_ONE_SECOND if model == TEACHER else H448]
        for c, _ in ALLOWANCE_CAUSES:
            rate = group[off][UNEXPLAINED]["causes"][c]["per_1000"]
            expected[(model, c)] = report["simulation"][model][off]["words"] * (rate or 0) / 1000
    below_one = all(v < 1 for v in expected.values())
    return [
        "### Where they disagree, and the likely reasons",
        "",
        "- **Most device flags are placement, not pronunciation.** Of the teacher's flags under",
        f"  `{off}`, {100 * explained:.0f}% fall on explained words: ayat the reader skipped",
        f"  ({with_skips} of the {len(included)} included sessions hold such a skip), words placed on a",
        "  twin, before the session's start or after its end. A simulated item starts on its first",
        "  word and holds one segment, so the engine has nothing to skip; the simulation cannot show",
        "  these.",
        "- **The overall rates.** With the explained words left out, under `allowances_on` the device",
        f"  flags {against(TEACHER_ONE_SECOND, TEACHER)} words per 1,000 for the teacher (1 s hop) and",
        f"  {against(H448, H448)} for `h448` (one session). The device is one reader on one phone; the",
        "  simulation is one word from each of 155 Tadabur reciters and their recordings.",
        "- **On the device the allowances change flags, on a few recurring words.** Switching the",
        "  three allowances off adds, over the teacher's sessions, unexplained flags for: "
        + "; ".join(added) + ".",
        f"  The logs are one reader reciting a few pages again and again (page {page} in {sessions}",
        "  sessions), so a word the model mishears for this reader recurs in every session that",
        "  reaches it. The simulation has one word per reciter and never repeats a word.",
        "- **The simulation is small for causes this rare.** The column *device rate × 155* is how",
        "  many of the simulation's words the device's rate would flag. For the allowance-sensitive",
        "  causes under `allowances_off` it is "
        + ", ".join(f"{_num(v, 2)} ({GROUP_TITLES[m]}, {CAUSE_TITLES[c]})" for (m, c), v in expected.items())
        + (": below one word each, so the simulation finding none is what the device's rates predict."
           if below_one else "."),
        "- **Other differences, not separated here.** The device decodes with 6-bit CoreML from the",
        "  phone's microphone; the simulation with PyTorch bf16 on Tadabur audio. Sessions before",
        "  2026-04-29 ran a 3 s hop (the simulation models the 1 s hop, hence the 1 s-hop rows in the",
        "  comparison). The logged overlap truncation is measured below.",
        "",
    ]


def _truncation_section(report: dict) -> list[str]:
    names = _scoring_names(report)
    device, sim = report["truncation"]["device"], report["truncation"]["simulation"]
    rows = [[f"`{n}`", str(device["scorings"][n]["flags_logged"]), str(device["scorings"][n]["flags_lengthened"]),
             str(device["scorings"][n]["changed"]),
             *(f"{sim[m][n]['flags_full']} → {sim[m][n]['flags_cut']} ({sim[m][n]['changed']} changed)"
               for m in MODELS)] for n in names]
    capped = sum(e["capped_overlaps"] for e in report["logs"] if e["exclusion"] is None)
    updates = sum(e["updates"] for e in report["logs"] if e["exclusion"] is None and e["protocol"] == "1 s hop")
    return [
        "## The logged overlap",
        "",
        f"The app logs each update's overlap cut to {OVERLAP_CAP} Characters",
        "(`unconfirmed.prefix(30)`, `FollowAlongManager+iOS.swift:744`), so a replay hands the engine",
        f"a shorter preview than the device saw: {capped} of the included 1 s-hop sessions' {updates}",
        f"updates sit exactly at the cap. Two tests of its effect:",
        "",
        f"- **The device logs, previews lengthened.** {device['previews_lengthened']} previews in "
        f"{device['logs']} sessions",
        "  also appear in the transcriber's own line, which keeps 40 Characters",
        "  (`RealtimeTranscriber+iOS.swift:642`); those sessions were replayed with the longer text.",
        "- **The simulation, overlaps cut.** #90's word sample regraded with every check's overlap cut",
        f"  to {OVERLAP_CAP} Characters (the simulation's own decodes are whole).",
        "",
        *_table(["scoring", "device flags, logged", "device flags, lengthened", "device words changed",
                 "simulation `h448`: flags full → cut", "simulation teacher: flags full → cut"], rows),
    ]


def _status(entry: dict) -> str:
    return "included" if entry["exclusion"] is None else f"excluded: {entry['exclusion']}"


def _inventory_section(report: dict) -> list[str]:
    rows = [[f"`{e['log']}`", e["model"] or "–", e["app_version"] or "not recorded", e["committed"],
             e["protocol"], _num(e["median_inference_ms"], 0), e["start"],
             ", ".join(map(str, e["pages"])) or "–", e["ayat"],
             "✓ / ✓" if all(e["parity"].values()) else " / ".join("✓" if v else "✗" for v in e["parity"].values()),
             _status(e)] for e in report["logs"]]
    purposes = [[f"`{e['log']}`", e["purpose"] + (f" **Excluded:** {e['exclusion_detail']}." if e["exclusion"] else ""),
                 "; ".join(e["sources"]), e["model_basis"] or "–"] for e in report["logs"]]
    return [
        "## The inventory",
        "",
        "Every non-generated log in Muraja's `tools/test_logs` at the pin. *Committed* is the date",
        "Muraja first committed the log, which bounds its capture; no log records its app version,",
        "so a version is given only where the docs name it. *Inference* is the median time the log",
        "records per inference (ms). *Start* is the log's Start line (word 0: the ayah, before its",
        "first word). *Ayat* are those the shipped-default replay graded. *Parity* is whether the",
        "replay reproduces the log's committed `baseline.json` at the pin and at `main`.",
        "",
        *_table(["log", "model", "app version", "committed", "protocol", "inference", "start", "pages",
                 "ayat", "parity", "status"], rows),
        "What each log tests, from its docs, and why its model is assigned:",
        "",
        *_table(["log", "purpose", "sources", "model basis"], purposes),
    ]


def _per_log_section(report: dict) -> list[str]:
    names = _scoring_names(report)
    rows = []
    for e in report["logs"]:
        if e["exclusion"] is not None:
            continue
        counts = e["counts"]
        rows.append([f"`{e['log']}`", e["model"], e["protocol"], str(counts[names[0]]["words"]),
                     *(f"{counts[n]['flags']} · {counts[n]['unexplained_flags']}" for n in names)])
    return [
        "## Per log",
        "",
        "Included logs: graded words under the first scoring, and per scoring the false flags over",
        "every graded word · with the explained words left out.",
        "",
        *_table(["log", "model", "protocol", "words", *(f"`{n}`" for n in names)], rows),
    ]


def _explained_section(report: dict) -> list[str]:
    flagged: dict[tuple[str, str, str], int] = defaultdict(int)
    for row in report["words"]:
        if row["explained"] and row["scoring"] == "shipped_default":
            flagged[(row["log"], row["explained"], row["basis"])] += 1
    rows = [[f"`{e['log']}`", f"{x['first']}–{x['last']}", x["kind"], x["basis"], x["source"]]
            for e in report["logs"] if e["exclusion"] is None for x in e["explained"]]
    by_basis = defaultdict(int)
    for (_, _, basis), n in flagged.items():
        by_basis[basis] += n
    return [
        "## Explained words",
        "",
        "Words a log's docs say were skipped, re-read or never recited (basis *docs*), or, where the",
        "docs name the session's purpose but not the words, its own decode and Start line show were",
        "never recited (basis *decode*: read from the log for this report; no log text is copied",
        "here). Under the shipped default they explain "
        + ", ".join(f"{n} flags by {basis}" for basis, n in sorted(by_basis.items())) + ".",
        "",
        *_table(["log", "words", "kind", "basis", "source"], rows),
    ]


def _mistakes_section(report: dict) -> list[str]:
    names = _scoring_names(report)
    logs = {e["log"] for e in report["logs"] if e["exclusion"] == DOCUMENTED_MISTAKE}
    models = Counter(e["model"] for e in report["logs"] if e["log"] in logs)
    lines = [
        "## For reference: the sessions excluded for a documented mistake",
        "",
        "These device sessions are out of every rate above: their docs record the reader slipping",
        f"into another passage's wording ({', '.join(f'{n} {GROUP_TITLES[m]}' for m, n in sorted(models.items()))}).",
        "Their flags are listed so that evidence is not lost, but they are no false-flag rate: the",
        "slip's own words and the engine's reaction to it are in them.",
        "",
    ]
    for model in MODELS:
        by = report["documented_mistakes"][model]
        rows = [[f"`{n}`", fmt_rate(by[n][ALL_WORDS]["false_flags"]),
                 ", ".join(f"{CAUSE_TITLES[c]} {by[n][ALL_WORDS]['causes'][c]['flags']}" for c in CAUSES
                           if by[n][ALL_WORDS]["causes"][c]["flags"]) or "–",
                 ", ".join(f"{k} {v}" for k, v in {**by[n][ALL_WORDS]["marks"], **by[n][ALL_WORDS]["pairs"]}.items())
                 or "–"] for n in names]
        lines += [f"**{GROUP_TITLES[model]}**", ""] + _table(["scoring", "flags / words", "causes", "detail"], rows)
    return lines + ["Their flagged words:", "", *_word_table(report, logs, explained=False)]


def render(report: dict) -> str:
    muraja = report["muraja"]
    included = [e for e in report["logs"] if e["exclusion"] is None]
    excluded = defaultdict(list)
    for e in report["logs"]:
        if e["exclusion"]:
            excluded[e["exclusion"]].append(e["log"])
    models = defaultdict(int)
    for e in included:
        models[e["model"]] += 1
    parity_ok = all(all(e["parity"].values()) for e in report["logs"])
    lines = [
        "# Device-log false flags: Muraja's own sessions, replayed (#151)",
        "",
        "Generated by `python -m training.device_logs` (from `tools/`); do not edit by hand. The",
        "machine-readable report, with every flagged word per scoring, is",
        "`tools/device_logs/report.json`; the hand-written inventory it reads is",
        "`tools/device_logs/inventory.json`. Numbers are reported as measured, with no",
        "recommendation on allowances or thresholds.",
        "",
        "Muraja's `tools/test_logs` hold real sessions from the device: every check the app ran,",
        "decoded on the phone. The owner (2026-10-14): they hold no intentional pronunciation or",
        "substitution mistake, and each has docs on what went wrong or what it tests. A word the app",
        "shows as not correct is therefore a **false flag** unless its log's docs say what else it",
        "is. Each log is replayed through Muraja's own replay (`tools/replay_log.swift`, by",
        "`tools/muraja_harness/log_replay.swift`) under every scoring, and every flag is attributed",
        "with `training/word_flags.py`.",
        "",
        "## Read this first",
        "",
        f"- **{len(report['logs'])} logs, {len(included)} included** "
        f"({', '.join(f'{n} {GROUP_TITLES[m]}' for m, n in sorted(models.items()))}). Excluded: "
        + "; ".join(f"{len(logs)} {EXCLUSION_TITLES[reason]}" for reason, logs in sorted(excluded.items())) + ".",
        f"- **{models[H448]} included session{'s' if models[H448] != 1 else ''} ran `h448`**; with one log a",
        "  rate has no interval. The other `h448` sessions are excluded by their docs for a slip into",
        "  another passage; their flags are listed at the end.",
        f"- **Parity holds{'' if parity_ok else ' only in part'}.** "
        + ("Every log's shipped-default replay reproduces its committed `baseline.json` (every word's "
           "quality and the final position) at the pin and at `main`." if parity_ok else
           "See the inventory's parity column."),
        f"- **Muraja `main` ({muraja['main'][:7]}) grades every log as the pin ({muraja['pin'][:7]}) does**"
        + (" under every scoring" if not muraja["main_differs_from_pin"] else
           f" except {', '.join(muraja['main_differs_from_pin'])}")
        + ": the two differ only outside the follow-along engine (voice search).",
        "- **A graded word stands for a recited word.** Explained words (skipped, re-read, never",
        "  recited) are reported both ways.",
        "",
        "## When `h448` shipped",
        "",
        H448_SHIP_EVIDENCE + _inference_evidence(report),
        *_scorings_section(report),
        *_rates_section(report),
        "## The unexplained false flags, word by word",
        "",
        "Every word of an included log that some scoring shows as not correct and no doc explains;",
        "each cell is the quality shown and its causes (the tashkeel marks and target pairs in",
        "brackets).",
        "",
        *_word_table(report, {e["log"] for e in included}, explained=False),
        *_comparison_section(report),
        *_truncation_section(report),
        *_per_log_section(report),
        *_explained_section(report),
        *_inventory_section(report),
        *_mistakes_section(report),
        "## Re-running",
        "",
        "```bash",
        "# from tools/, on a Mac: build the harness at the pin and at Muraja main, then report",
        "bash muraja_harness/build.sh",
        f"bash muraja_harness/build.sh {muraja['main']}",
        "python -m training.device_logs",
        "```",
        "",
        "Harness builds: " + "; ".join(f"{name} `{b['muraja_commit'][:7]}` ({b['compiler']})"
                                       for name, b in muraja["builds"].items()) + ".",
        "",
    ]
    return "\n".join(lines)

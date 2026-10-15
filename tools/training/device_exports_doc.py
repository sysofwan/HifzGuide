"""Render the owner's ``h448`` session report (``docs/owner-device-sessions.md``) from its JSON.

Presentation only: every number comes from :func:`training.device_exports.build_report`, and the
doc is regenerated with it, never edited by hand.
"""

from __future__ import annotations

from collections import defaultdict

from training.device_exports import AS_RECORDED, IN_MISTAKE_AYAH, MISTAKE, MODEL_ERROR, MODES
from training.device_logs import (
    ALL_WORDS,
    H448,
    SLIP_WIDTHS,
    TEACHER,
    TEACHER_ONE_SECOND,
    UNEXPLAINED,
    position,
    sensitivity,
)
from training.device_logs_doc import CAUSE_TITLES, flag_cell, fmt_rate, md_table, num
from training.word_flags import CAUSES

#: Where each population of the allowance table comes from (``None``: every word).
WIDTH_TITLES = {"None": "these sessions", "narrow": "added phrases excluded (narrow)",
                "wide": "added phrases excluded (wide)"}
ERA_TITLES = {TEACHER_ONE_SECOND: "#151 teacher, 1 s hop", TEACHER: "#151 teacher, all eras",
              H448: "#151 `h448` sessions"}


def _names(report: dict) -> list[str]:
    return [s["name"] for s in report["scorings"]]


def _targets(report: dict) -> list[dict]:
    return [s for s in report["sessions"] if s["target"]]


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _cell_words(rows: list[tuple[str, int]]) -> str:
    return ", ".join(f"{w} ({n} sessions)" if n > 1 else w for w, n in rows) or "–"


# --- sections -----------------------------------------------------------------------------------------
def _exports_section(report: dict) -> list[str]:
    rows = [[f"`{e['name']}`", f"`{e['sha256']}`", e["mode_at_export"], ", ".join(e["sessions"])]
            for e in report["exports"]]
    sessions = []
    for s in report["sessions"]:
        status = ("**target**" if s["target"] else
                  f"the tail of {s['copy_of']} (its opening truncated by the export)" if s["copy_of"] else
                  "partial, not replayed")
        sessions.append([s["session"], str(s["page"] or "–"), s["kind"] or "–", s["app_version"] or "–",
                         f"`{s['model']}`" if s["model"] else "–", s["run_basis"] or "–", status])
    return [
        "## The exports and their sessions",
        "",
        "The exports stay outside the repo. They are pinned here by name and SHA-256, and the",
        "generator refuses any other file. *Mode at export* is the Pronunciation setting the export",
        "records; no log line records the mode a session ran in.",
        "",
        *md_table(["export", "SHA-256", "mode at export", "sessions"], rows),
        "A session opens at ScoreCollector's *session pending on page N* line. The export keeps the",
        "tail of the app's log, so it can begin inside an earlier session. Such a session is matched",
        "by its opening lines to the whole session in another export, and it is not replayed. The",
        "app version and model are read from the app run's header and its model-load line. Where",
        "the export truncated those, they are read from the same app run in another export.",
        "",
        *md_table(["session", "page", "kind", "app", "model", "model read from", "status"], sessions),
        "No export holds a session on another app version or model. The 1.0.26 app run that",
        "export 69 names as its previous session holds no log lines.",
        "",
    ]


def _sessions_section(report: dict) -> list[str]:
    rows = [[s["session"], str(s["page"]), f"**{s['mode']}**",
             next(e["mode_at_export"] for e in report["exports"] if e["name"] == s["export"]),
             s["start"], s["ayat"], f"{s['words']} / {s['ayat_words']}", str(s["skipped"]), str(s["updates"]),
             str(s["capped_overlaps"]),
             str(s["previews_lengthened"]), num(s["median_inference_ms"], 0)]
            for s in _targets(report)]
    return [
        "## The target sessions",
        "",
        "*Mode* is the mode whose replay reproduces the device's grade decisions (see *Parity*).",
        "*Start* is where Muraja's replay seats the engine: the ayah of the session's first",
        "`expected=`, word 0. The app stopped logging a `Start:` line in Muraja `ba91ac3`.",
        "*Words* are the words the session kept a grade for, of the words in its ayat; *skipped*",
        "those graded skipped. *Capped* counts updates whose logged",
        "overlap sits at the 30-Character cap. *Lengthened* counts previews replayed with the",
        "transcriber's own 40-Character line. *Inference* is the median per inference (ms); the",
        "teacher's sessions in #151 all ran far slower.",
        "",
        *md_table(["session", "page", "mode", "mode at export", "start", "ayat", "words", "skipped", "updates",
                   "capped", "lengthened", "inference"], rows),
    ]


def _parity_section(report: dict) -> list[str]:
    rows = []
    for s in _targets(report):
        for run, by_mode in s["parity"].items():
            cells = []
            for mode in MODES:
                p = by_mode[mode]
                mark = " ✓" if p["exact"] else f", {_plural(len(p['differing_words']), 'word')} differ"
                cells.append(f"{p['matched']} / {p['device']}{mark}")
            rows.append([s["session"], f"`{run}`", *cells])
    logged = []
    for s in _targets(report):
        p = s["parity"]["pin/logged"][s["mode"]]
        logged.append([s["session"], s["mode"], str(p["device"] - p["matched"]), ", ".join(p["differing_words"]) or "–"])
    commits = []
    for s in _targets(report):
        c = s["commit_parity"]
        later = ", ".join(f"{x['word']} ({x['committed']} → {x['kept']})" for x in c["regraded_after_commit"]) or "–"
        other = ", ".join(f"{x['word']} ({x['committed']} → {x['kept']})" for x in c["unexplained"]) or "none"
        commits.append([s["session"], str(c["committed"]), str(c["kept"]), str(c["equal"]), later, other])
    pin, main = report["muraja"]["pin"][:7], report["muraja"]["main"][:7]
    return [
        "## Parity with the device",
        "",
        "The word scorer logs every end-word grade it decides (`Grade end word …`,",
        "`QuranFollowAlong+WordScoring.swift:750`). The harness keeps the same lines from the replay",
        "(`muraja_harness/log_replay.swift`). Each cell gives the device's decisions that the replay",
        "reproduces, in order (longest common subsequence), out of the device's total. ✓ means the",
        "two sequences are identical. Each session is replayed in every mode with the shipped flags,",
        f"at the pin `{pin}` and at Muraja `main` `{main}`, from its log as logged and with each",
        "preview lengthened. App 1.0.27 was built from neither commit exactly. The exports log",
        "voice search's *the Arabic speech recognizer is available* line, which Muraja first logs in",
        "`ca4d665`, after the 1.0.27 version bump `99c326f`. The two commits differ only in voice",
        "search.",
        "",
        *md_table(["session", "replay", *(f"`{m}`" for m in MODES)], rows),
        "**Replayed as logged, no mode reproduces the device.** The app logs each update's overlap",
        "cut to 30 Characters (`FollowAlongManager+iOS.swift:744`). For a preview, the",
        "transcriber's own line keeps 40 (`RealtimeTranscriber+iOS.swift:642`;",
        "`device_logs.lengthened_previews`). **With the previews lengthened, exactly one mode",
        "reproduces every grade decision the device logged for every word, at both commits.**",
        "That mode is each session's mode. All reported numbers below come from that replay at the",
        "pin. The decisions the as-logged replay gets wrong in the session's own mode:",
        "",
        *md_table(["session", "mode", "decisions not reproduced", "words"], logged),
        "**Committed grades.** ScoreCollector logs each word's grade as it commits it (`Commit …`).",
        "This covers every word the session grades, not only end words. The replay keeps the same",
        "grade for every committed word except those the device regraded after committing them",
        "(an `Upgraded:` line for the word later in the log).",
        "",
        *md_table(["session", "committed", "kept by the replay", "equal", "regraded after the commit",
                   "other differences"], commits),
    ]


def _scorings_section(report: dict) -> list[str]:
    flag = {None: "the mode's", True: "on", False: "**off**"}
    rows = [[f"`{s['name']}`", f"`{s['mode']}`" if s["name"] != AS_RECORDED else "the session's own",
             flag[s["soft_pairs_enabled"]], flag[s["shaddah_suppression"]],
             "shown" if s["tashkeel_shown"] else "hidden"] for s in report["scorings"]]
    return [
        "## The scorings",
        "",
        "#151's scorings, all in Balanced, plus the session as the device ran it (`as_recorded`:",
        "its own mode, the shipped flags, tashkeel shown, as the exports' settings have it). Strict",
        "forgives no soft pair. Each soft pair is also switched off alone (*Flags added when one",
        "allowance is off*).",
        "",
        *md_table(["scoring", "mode", "soft-pair forgiveness", "shaddah suppression", "tashkeel"], rows),
    ]


def _rates_section(report: dict) -> list[str]:
    names = _names(report)
    rates, slips = report["rates"], report["slips_excluded"]
    rows = [[f"`{n}`", fmt_rate(rates[n][ALL_WORDS]["false_flags"]), fmt_rate(rates[n][UNEXPLAINED]["false_flags"]),
             *(fmt_rate(slips[w][n]["false_flags"]) for w in SLIP_WIDTHS)] for n in names]
    per_session = [[s["session"], str(s["page"]), *(f"{s['counts'][n]['unexplained_flags']} / "
                                                     f"{s['counts'][n]['unexplained_words']}" for n in names)]
                   for s in _targets(report)]
    causes = [[CAUSE_TITLES[c], *(str(rates[n][UNEXPLAINED]["causes"][c]["flags"]) for n in names)] for c in CAUSES]
    details = sorted({key for n in names for kind in ("marks", "pairs") for key in rates[n][UNEXPLAINED][kind]})
    causes += [[f"— {key}", *(str({**rates[n][UNEXPLAINED]["marks"], **rates[n][UNEXPLAINED]["pairs"]}.get(key, 0))
                              for n in names)] for key in details]
    return [
        "## False flags per 1,000 graded words",
        "",
        "Count / graded words · per 1,000. With three sessions, no interval is admissible (n/c);",
        "#151's interval resamples sessions. *Owner's mistakes left out* drops from both counts the",
        "word the owner names as a mistake (2:60:11) and every word of 2:256, the ayah the owner",
        "says holds mistakes without naming its words (*The owner's mistakes*). This is the",
        "false-flag rate. No other word is",
        "explained. #151's rule admits only decode evidence of a word skipped or never recited, and",
        "each session grades every word of its ayat, none as skipped. A few words the reader repeats",
        "inside a session stay in, as #151 keeps the re-reads no doc records. The added-phrase",
        "columns also leave out #151's slip region around each added phrase",
        "(`device_logs.regions`): narrow runs from the phrase to the end of its ayah, wide covers",
        "its ayah and the next; both add the passage whose wording it was, in full.",
        "",
        *md_table(["scoring", "every graded word", "owner's mistakes left out", "added phrases excluded (narrow)",
                   "added phrases excluded (wide)"], rows),
        "Per session, owner's mistakes left out (flags / words):",
        "",
        *md_table(["session", "page", *(f"`{n}`" for n in names)], per_session),
        "Flags per cause, owner's mistakes left out (a word may carry several causes; the tashkeel",
        "marks and pairs behind them below):",
        "",
        *md_table(["", *(f"`{n}`" for n in names)], causes),
    ]


def _allowances_section(report: dict) -> list[str]:
    table = report["allowances"]

    def cell(c: dict) -> str:
        if not c["allowance"]:
            return f"no allowance; {c['flags']} flagged under `allowances_on`"
        return fmt_rate(c) + (f", {c['removed']} removed" if c["removed"] else "")

    widths = list(WIDTH_TITLES)
    rows = [[label, *(cell(by[w]) for w in widths)] for label, by in table.items()]
    carriers = [[label, _cell_words(by["None"]["carriers"])] for label, by in table.items() if by["None"]["carriers"]]
    return [
        "## Flags added when one allowance is off",
        "",
        "As #151 computes it: information for the owner's per-allowance decisions (#128), not a",
        "verdict. Each cell counts the words flagged with the allowance off that `allowances_on`",
        "does not flag, over the words graded with it off, per 1,000. Owner's mistakes are left",
        "out. The soft pairs are switched off one at a time, each through the harness's one-pair",
        "switch. *Removed* counts words that `allowances_on` flags and the scoring does not.",
        "",
        *md_table(["allowance", *(WIDTH_TITLES[w] for w in widths)], rows),
        "The words carrying them (these sessions):",
        "",
        *(md_table(["allowance", "words"], carriers) if carriers else ["None.", ""]),
    ]


def _status(verdict: dict) -> str:
    """A non-correct word's status as the tables show it."""
    if verdict["status"] == MISTAKE:
        return "the owner's mistake"
    if verdict["status"] == IN_MISTAKE_AYAH:
        return "flagged in an ayah the owner says holds mistakes (words not identified)"
    if verdict["status"] == MODEL_ERROR:
        return f"false flag: model error (owner {verdict['date']}; {verdict['explained_by']})"
    return "false flag: assumed (never shown on the phone)"


def _phone_section(report: dict) -> list[str]:
    rows = []
    for s in _targets(report):
        kept = {r["word"]: r for r in report["words"] if r["log"] == s["session"] and r["scoring"] == AS_RECORDED}
        for c in s["committed_flags"]:
            k = kept.get(c["word"])
            rows.append([s["session"], c["word"], f"{c['quality']} ({', '.join(c['errors'])})",
                         flag_cell(k) if k else "correct", _status(c["verdict"]),
                         c["verdict"].get("evidence", "–")])
    statements = [[st["date"], f"\"{st['text']}\"", st["source"]] for st in report["statements"]]
    return [
        "## What the phone showed, and the owner's verdicts",
        "",
        "The owner gave verdicts on the words the phone's ScoreCollector committed as not correct",
        "(its `Commit` lines). A committed grade can differ from the grade the session finally kept,",
        "the one every rate here uses (#151's rule). The device regrades a word after committing",
        "it when a later decode ranks higher (`GradeStore`'s ratchet). *Explained by*",
        "comes from reading the decode at each word: **decode** means the model heard another letter",
        "or mark there; **boundary** means a window ended inside the word and the engine graded it",
        "on a decode that lost part of it. None is an alignment effect: each word's errors match",
        "the decode at that word.",
        "",
        *md_table(["session", "word", "committed (errors)", "kept (as recorded)", "status", "explained by"], rows),
        "The owner's statements:",
        "",
        *md_table(["date", "statement", "source"], statements),
    ]


def _words_section(report: dict) -> list[str]:
    names = _names(report)
    cells: dict[tuple[str, str], dict[str, dict]] = defaultdict(dict)
    for row in report["words"]:
        cells[(row["log"], row["word"])][row["scoring"]] = row
    rows = []
    for (session, word), by in sorted(cells.items(), key=lambda kv: (kv[0][0], position(kv[0][1]))):
        rows.append([session, word, _status(next(iter(by.values()))["verdict"]), *(flag_cell(by.get(n)) for n in names)])
    return [
        "## Every non-correct word, under every scoring",
        "",
        "Every word that some scoring shows as not correct in the grade the session kept. Each cell",
        "gives the quality shown and its causes, with the tashkeel marks and pairs in brackets.",
        "Every word the phone showed as not correct has the owner's verdict. The phone never showed",
        "a word flagged only under another scoring, so the owner had nothing to judge there. Such a",
        "word is a false flag on the rule that the recitation is correct. Words flagged only with",
        "one soft pair off are listed under *Flags added when one allowance is off*; each is also",
        "flagged under `soft_pair_forgiveness_off` here.",
        "",
        *md_table(["session", "word", "status", *(f"`{n}`" for n in names)], rows),
    ]


def _catch_section(report: dict) -> list[str]:
    names = _names(report)
    catch = report["catch"]

    def causes(cell: dict) -> str:
        return ", ".join(f"{CAUSE_TITLES[c]} {k}" for c, k in cell["causes"].items()) or "–"

    def flagged(cell: dict) -> str:
        return ", ".join(f"{w['word']} ({w['quality']})" for w in cell["flagged_words"]) or "–"

    words = [[f"`{n}`", f"{catch[n]['words']['flagged']} / {catch[n]['words']['graded']}",
              causes(catch[n]["words"]), flagged(catch[n]["words"])] for n in names]
    ayat = [[f"`{n}`", str(catch[n]["ayat"]["flagged"]), causes(catch[n]["ayat"]), flagged(catch[n]["ayat"])]
            for n in names]
    added = [[s["session"], f"after {a['start']}" if a["start"] == a["end"] else f"between {a['start']} and {a['end']}",
              a["twin"], f"#{a['detection']}: {' → '.join(a['kinds'])}",
              "yes" if a["saved"] else "no (the export holds no end of this session)",
              ", ".join(f"{w} {q}" for w, q in a["kept"].items())]
             for s in _targets(report) for a in s["added"]]
    def spans(kind: str) -> str:
        return "; ".join(f"{s['session']}: " + ", ".join(m["first"] if m["first"] == m["last"] else f"{m['first']}–{m['last']}"
                                                        for m in s["mistakes"] if m["kind"] == kind)
                         for s in _targets(report) if any(m["kind"] == kind for m in s["mistakes"]))

    return [
        "## The owner's mistakes: first device evidence on the catch side",
        "",
        "What the owner says they got wrong is not a false flag. It is out of every false-flag rate",
        "above and is reported here, apart from the rates, in three kinds.",
        "",
        f"**Words the owner names as mistakes** ({spans('mistake')}). Each scoring's count of them",
        "flagged by the app, out of those graded:",
        "",
        *md_table(["scoring", "flagged / graded", "causes", "words"], words),
        f"**An ayah the owner says holds mistakes** ({spans('mistake_ayah')}). The owner did not say",
        "which words were wrong, so no word of it is a known mistake and there is no catch rate.",
        "These are the flags the app showed in it, with their causes. Each is a flag *in an ayah",
        "that holds mistakes*, not a confirmed catch:",
        "",
        *md_table(["scoring", "flags in the ayah", "causes", "words"], ayat),
        "**Added words.** Muraja's shadow-speech detector classified two stretches of speech as",
        "added. The owner says the added words were a mistake. The passage whose wording each one",
        "is was read from the decode. It is not quoted here: the log text stays private. An added",
        "phrase is not a mushaf word, so the app grades no word for it. What the engine did:",
        "",
        *md_table(["session", "placed", "wording of", "detection (its classifications)", "saved as added words",
                   "grades kept on its words (as recorded)"], added),
    ]


def _substitutions_section(report: dict) -> list[str]:
    sub = report["substitutions"]
    scoring = "shipped_default"
    columns = {
        "these sessions": sub["sessions"][scoring],
        "#151 teacher, 1 s hop": sub["device_logs"][TEACHER_ONE_SECOND][scoring],
        "#151 teacher, all eras": sub["device_logs"][TEACHER][scoring],
        "#151 `h448`, narrow": sub["device_logs"][sensitivity(H448, "narrow")][scoring],
        "#90 `h448/checks`": sub["simulation"][H448][scoring],
        "#90 `base/checks`": sub["simulation"][TEACHER][scoring],
        "owner's mistakes (catch side)": sub["owner_mistakes"][scoring],
        "in 2:256 (mistakes, words not identified)": sub["mistake_ayat"][scoring],
    }

    def table(field: str, key: str) -> list[str]:
        keys = sorted({(t["cause"], t[key]) for tally in columns.values() for t in tally[field]},
                      key=lambda ck: (-sum(next((t["words"] for t in tally[field] if (t["cause"], t[key]) == ck), 0)
                                           for tally in list(columns.values())[:-2]), ck))
        rows = [[f"{label} ({CAUSE_TITLES[cause]})",
                 *(str(next((t["words"] for t in tally[field] if (t["cause"], t[key]) == (cause, label)), 0) or "·")
                   for tally in columns.values())] for cause, label in keys]
        return md_table(["", *columns], [["flags", *(str(t["flags"]) for t in columns.values())], *rows])

    def leaders(tally: dict, field: str, key: str) -> str:
        top = max((t["words"] for t in tally[field]), default=0)
        return ", ".join(t[key] for t in tally[field] if t["words"] == top) + f" ({top} of {tally['flags']} flags)" if top else "none"

    lead = [f"- {title}: substitution {leaders(t, 'substitutions', 'substitution')}; letter "
            f"{leaders(t, 'expected', 'expected')}" for title, t in list(columns.items())[:-2]]
    return [
        "## The consonant flags, by substitution",
        "",
        "Every consonant error behind a consonant flag: `expected→heard`, or `expected→∅` for a",
        "deletion. The tally is `word_flags.substitution_tally` over the grade each word kept. Each",
        "cell counts flagged words, under `shipped_default`, explained words and the owner's",
        "mistakes left out. A word counts once per substitution, and a word with several errors",
        "counts under each. *Flags* counts every false flag of the population, tashkeel ones",
        "included. #151's device logs are replayed here at the pin, and each population's flag",
        "count is checked against #151's report. #90's errors are read from its committed report.",
        "The last two columns are the catch side, kept apart: the word the owner names as a",
        "mistake, and the flags in the ayah the owner says holds mistakes.",
        "",
        *table("substitutions", "substitution"),
        "By the expected letter (a word counts once per letter):",
        "",
        *table("expected", "expected"),
        "The most frequent substitution and expected letter in each population (ties listed):",
        "",
        *lead,
        "",
    ]


def _recurring_section(report: dict) -> list[str]:
    names = _names(report)
    rows = []
    for item in report["recurring"]:
        now = []
        for s in item["now"]:
            decisions = ", ".join(f"{q} {n}" for q, n in s["decisions"].items())
            flagged = "; ".join(f"`{n}` {g['quality']} ({', '.join(g['causes'] + g['pairs'] + g['marks'])})"
                                for n, g in s["flagged"].items()) or "correct under every scoring"
            now.append(f"{s['session']}: {flagged}. Device end-word decisions: {decisions}")
        era = []
        for group, title in ERA_TITLES.items():
            e = item["era"][group]
            flagged = ", ".join(f"`{n}` {k}" for n, k in e["flagged"].items() if k)
            era.append(f"{title}: {_plural(e['sessions'], 'session')}" + (f"; flagged {flagged}" if flagged else
                                                              "; flagged under no scoring" if e["sessions"] else ""))
        rows.append([item["word"], "<br>".join(now) or "not graded", "<br>".join(era)])
    sample = ", ".join(f"{s['ayah']} (reference word {s['reference_word']}, {s['verdict'] or 'no verdict'})"
                       for s in report["simulation_sample"])
    return [
        "## The recurring words, then and now",
        "",
        "The words that carried #151's allowance-sensitive flags. *Now*: each target session that",
        "graded the word, the scorings that show it as not correct, and the device's own end-word",
        "decisions for it. A word locks correct once one decision is correct (`GradeStore`'s upward",
        "ratchet), so other decisions there do not show. *Then*: #151's sessions that graded the",
        "word's ayah, and in how many of them each scoring showed the word as not correct",
        f"(explained and slip-region words left out). The scorings: {', '.join(f'`{n}`' for n in names[:-1])}.",
        "",
        *md_table(["word", "now (`h448`, 1.0.27)", "then (#151)"], rows),
        f"#90's simulated word sample holds none of these words. On these sessions' ayat it holds {sample}",
        "(numbered by the sample's own reference words).",
        "",
    ]


def _comparison_section(report: dict) -> list[str]:
    comparison = report["comparison"]
    rates = report["rates"]
    narrow = report["slips_excluded"]["narrow"]
    rows = []
    for s in report["scorings"]:
        n = s["name"]
        theirs = comparison["rates"].get(n)
        sim = theirs and theirs["simulation"]
        rows.append([f"`{n}`", fmt_rate(rates[n][UNEXPLAINED]["false_flags"]), fmt_rate(narrow[n]["false_flags"]),
                     *((fmt_rate(theirs[TEACHER_ONE_SECOND]), *(fmt_rate(theirs[sensitivity(H448, w)]) for w in SLIP_WIDTHS),
                        f"{sim['flags']} / {sim['words']} (weighted {num(sim['per_1000'])} [{num(sim['lower'])}, {num(sim['upper'])}])")
                       if theirs else ("–",) * 4)])
    allowance_rows = []
    for label, by in report["allowances"].items():
        theirs = comparison["allowances"][label]
        mine = by["None"]
        render = (lambda c: fmt_rate(c)) if mine["allowance"] else (lambda c: f"{c['flags']} flagged")
        allowance_rows.append([label, render(mine), render(theirs[TEACHER_ONE_SECOND]),
                               *(render(theirs[sensitivity(H448, w)]) for w in SLIP_WIDTHS)])
    return [
        "## Against #151 and #90",
        "",
        "These sessions (owner's mistakes left out, and with the added-phrase regions excluded,",
        "narrow) against #151's teacher sessions on the 1 s hop and its `h448` sensitivity (all",
        "`h448` sessions, slip regions excluded), and against #90's simulated `h448/checks` word",
        "sample (correctly recited words flagged, of 155). #151's rows are read from",
        "`tools/device_logs/report.json`. Everything is per 1,000 graded words.",
        "",
        *md_table(["scoring", "these sessions", "added phrases excluded (narrow)", "#151 teacher, 1 s hop",
                   "#151 `h448`, narrow", "#151 `h448`, wide", "#90 `h448/checks`"], rows),
        "Flags added when one allowance is off, against #151:",
        "",
        *md_table(["allowance", "these sessions", "#151 teacher, 1 s hop", "#151 `h448`, narrow",
                   "#151 `h448`, wide"], allowance_rows),
    ]


def _summary(report: dict) -> list[str]:
    targets = _targets(report)
    rates = report["rates"]
    exact = all(s["parity"][run][s["mode"]]["exact"] for s in targets for run in ("pin/lengthened", "main/lengthened"))
    sessions = "; ".join(f"page {s['page']} ({s['session']}, {s['mode']}, {s['words']} words)" for s in targets)
    as_logged = sum(s["parity"]["pin/logged"][s["mode"]]["device"] - s["parity"]["pin/logged"][s["mode"]]["matched"]
                    for s in targets)
    decisions = sum(s["parity"]["pin/logged"][s["mode"]]["device"] for s in targets)
    headline = ", ".join(f"`{n}` {num(rates[n][UNEXPLAINED]['false_flags']['per_1000'])}"
                         f" ({rates[n][UNEXPLAINED]['false_flags']['flags']} / {rates[n][UNEXPLAINED]['false_flags']['words']})"
                         for n in (AS_RECORDED, "shipped_default", "allowances_on", "allowances_off"))
    catch = report["catch"][AS_RECORDED]
    shown = [c for t in targets for c in t["committed_flags"]]
    errors = [c["verdict"] for c in shown if c["verdict"]["status"] == MODEL_ERROR]
    by = defaultdict(int)
    for v in errors:
        by[v["explained_by"]] += 1
    mistakes = sum(c["verdict"]["status"] == MISTAKE for c in shown)
    in_ayat = sum(c["verdict"]["status"] == IN_MISTAKE_AYAH for c in shown)
    verdicts = (f"of {len(shown)} words, {len(errors)} are model errors "
                f"({', '.join(f'{n} {how}' for how, n in sorted(by.items()))}), {mistakes} the owner's mistake "
                f"and {in_ayat} in the ayah the owner says holds mistakes")
    return [
        "## Read this first",
        "",
        f"- **{len(targets)} target sessions, all `h448` on app 1.0.27:** {sessions}. The other sessions",
        "  in the exports are truncated tails of these.",
        f"- **Parity {'is exact' if exact else 'is not exact'}.** With each preview lengthened to the transcriber's",
        "  own 40-Character line, the replay reproduces every grade decision the device logged, in",
        "  order, at the pin and at `main`. It does so in one mode per session, the mode reported.",
        f"  Replayed as logged, with each overlap cut as the app logs it, it misses {as_logged} of {decisions}",
        "  decisions. The kept grades also equal the device's committed grades, except where the",
        "  device regraded a word after committing it.",
        f"- **False flags per 1,000 graded words, owner's mistakes left out:** {headline}.",
        "  Three sessions give no interval.",
        f"- **Every word the phone showed as not correct has the owner's verdict:** {verdicts}. Words",
        "  flagged only under other scorings were never shown on the phone; they stay false flags, on",
        "  the rule that the recitation is correct.",
        f"- **The owner's mistakes (catch side):** as recorded, the app flagged {catch['words']['flagged']} of"
        f" {_plural(catch['words']['graded'], 'word')} the owner names",
        f"  ({', '.join(w['word'] for w in catch['words']['flagged_words']) or 'none'}). In 2:256, which the owner says"
        f" holds mistakes without naming the words, it showed {_plural(catch['ayat']['flagged'], 'flag')}",
        f"  ({', '.join(w['word'] for w in catch['ayat']['flagged_words']) or 'none'}); with the wrong words unknown,"
        " that is no catch rate.",
        "  The detector classified both added phrases as added. One was saved as added words; the",
        "  other session's end is not in the export.",
        "- No recommendation on allowances or thresholds.",
        "",
    ]


def render(report: dict) -> str:
    muraja = report["muraja"]
    return "\n".join([
        "# The owner's `h448` sessions, pages 4, 9 and 42, replayed (#153)",
        "",
        "Generated by `python -m training.device_exports <exports>` (from `tools/`); do not edit by",
        "hand. The machine-readable report is `tools/device_logs/owner_report.json`. The",
        "hand-written annotations it reads (pinned exports, the owner's statements and verdicts, the",
        "added phrases) are in `tools/device_logs/owner_sessions.json`. The method is #151's",
        "(`docs/device-log-false-flags.md`). The exports are the owner's private data: no log text,",
        "device info or settings dump is written here.",
        "",
        "The owner recorded these sessions on the app as shipped to measure false flags. A word the",
        "app shows as not correct is therefore a **false flag**, unless the owner said it was a",
        "mistake. The owner gave a verdict on every word the phone showed as not correct.",
        "",
        *_summary(report),
        *_exports_section(report),
        *_sessions_section(report),
        *_parity_section(report),
        *_scorings_section(report),
        *_rates_section(report),
        *_phone_section(report),
        *_words_section(report),
        *_allowances_section(report),
        *_catch_section(report),
        *_substitutions_section(report),
        *_recurring_section(report),
        *_comparison_section(report),
        "## Re-running",
        "",
        "```bash",
        "# from tools/, on a Mac: build the harness at the pin and at Muraja main, then report",
        "bash muraja_harness/build.sh",
        f"bash muraja_harness/build.sh {muraja['main']}",
        "python -m training.device_exports ~/Downloads/muraja-bug-report\\ {69,70,71}.txt",
        "```",
        "",
        "Harness builds: " + "; ".join(f"{name} `{b['muraja_commit'][:7]}` ({b['compiler']})"
                                       for name, b in muraja["builds"].items()) + ".",
        "",
    ])

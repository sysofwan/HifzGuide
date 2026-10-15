"""Render ``docs/qud-pilot.md`` from the pilot's report and the overlap check (#144).

Every number is read from ``tadabur/qud_eval/pilot/report.json`` and the overlap check's files
(``tadabur/qud_eval/overlap_*.json``); the prose states only what their construction fixes.
Nothing here recommends a threshold, a margin or a listening budget.
"""

from __future__ import annotations

import json

from tadabur.qud_overlap import PILOT, RESULTS_PATH, RULES_PATH, SCREEN_PATH
from training.carrier_windowing_doc import rate_cell
from training.direction_table import FAMILIES, GEMINATE, MUSHAF, OTHER, SWAP

#: The guard's floor #144 sets the per-pair counts against (acceptance rules, n >= 189).
FLOOR = 189
ARMS = ("base/spans", "base/stream_b0", "base/checks", "h448/spans", "h448/stream_b0",
        "h448/checks")


def _pct(x) -> str:
    return "–" if x is None else f"{100 * x:.1f}%"


def _overlap() -> list[str]:
    rules = json.loads(RULES_PATH.read_text(encoding="utf-8"))["lines"]
    screen = json.loads(SCREEN_PATH.read_text(encoding="utf-8"))
    results = json.loads(RESULTS_PATH.read_text(encoding="utf-8"))
    fp, timing, agg = rules["fingerprint"], rules["timing"], rules["aggregate"]
    out = [
        "## Overlap with Tadabur and Muaalem",
        "",
        "Checked by audio (`tadabur.qud_overlap`). The controls were committed before any pilot "
        "recording met Tadabur (`qud_eval/overlap_controls.json`) and the lines fixed from "
        "their scores (`overlap_rules.json`); the module docstring records the two amendments "
        "and when each was made.",
        "",
        f"- **Fingerprint line** (bit error rate at the best offset): {fp['line']:.3f}, halfway "
        f"between the highest of {fp['synthetic_positives']} synthetic positives (Tadabur clips "
        f"re-encoded as QUD-style MP3 and re-cut: {fp['synthetic_positive_max']:.3f}) and the "
        f"lowest of {fp['hard_negatives']} hard negatives (one reciter, one ayah, two "
        f"recordings: {fp['hard_negative_min']:.3f}).",
        f"- **Timing line**: {timing['line']} s, 1.5 times the largest timing distance of the "
        f"{timing['positives_timed']} timed cross-aligner pairs the fingerprint confirms as one "
        f"recording ({timing['positives']} in all, over "
        f"{len(timing['fingerprint_matches_by_reciter'])} Tadabur reciters). On the controls it "
        f"passes {timing['same_reciter_negatives_passing']} of "
        f"{timing['same_reciter_negatives_timed']} same-reciter negatives and "
        f"{timing['other_reciter_negatives_passing']} of {timing['other_reciter_negatives_timed']} "
        "other-reciter negatives.",
        f"- **Per-reciter test** (stage 1, reported only): an id is flagged when its pairs within "
        f"the line are improbable at the negatives' pass rate (Wilson upper "
        f"{agg['q_upper']}, level {agg['alpha']}); on the controls it flags "
        f"{agg['control_overlaps_flagged']} of {agg['control_overlaps']} reciter-mushaf overlaps "
        f"and {agg['control_non_overlaps_flagged']} of {agg['control_non_overlaps']} others.",
        "",
        "| mushaf | ayat compared (stage 1) | pairs timed | ids flagged by the per-reciter "
        "test | stage 2 pairs | fingerprinted | sealed, not read | matches detected | lowest BER |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for slug in PILOT:
        s, r = screen["mushafs"][slug], results["tadabur"][slug]
        out.append(f"| `{slug}` | {s['ayat_compared']} of {s['ayat']} | {s['pairs']} | "
                   f"{len(s['flagged_reciters'])} | {r['pairs']} | {r['fingerprinted']} | "
                   f"{r['sealed_not_compared']} | {len(r['matches'])} | {r['lowest_ber']} |")
    neanea = [c for c in json.loads(RULES_PATH.read_text(encoding="utf-8"))["scores"]
              ["cross_aligner"] if c["tadabur_reciter"] == 18 and c["slug"] == "ahmed_nuayna_qdc"]
    timed = [c for c in neanea if c["timing_distance"] is not None]
    zone = {slug: results["tadabur"][slug]["between_line_and_hard_negatives"] for slug in PILOT}
    out += [
        "",
        "Stage 2 fingerprinted every pair within the timing line and every row of the pilot "
        "reciter's named Tadabur id (Almiman: 93) with a pilot ayah, timed or not (" + "; ".join(
            f"`{slug}` {results['tadabur'][slug]['named_pairs']} named-id pairs, "
            f"{results['tadabur'][slug]['named_pairs_untimed']} of them untimed"
            for slug in PILOT if results["tadabur"][slug]["named_ids"]) + "). Pairs "
        "between the fingerprint line and the lowest hard negative: " + "; ".join(
            f"`{slug}` {len(z)} (BER {min(x['ber'] for x in z):.3f} to "
            f"{max(x['ber'] for x in z):.3f})" if z else f"`{slug}` none"
            for slug, z in zone.items()) + ", every one far above the line and above every "
        "control positive.",
        "",
        "**What zero matches means: no overlap detected under this screen, not proven "
        "absence.** The controls show the fingerprint can miss a copy of one recording. Tadabur "
        f"reciter 18's {len(neanea)} clips against QUD's `ahmed_nuayna_qdc` (the same reciter) "
        f"match the fingerprint line {sum(c['ber'] <= fp['line'] for c in neanea)} time "
        f"(BER {min(c['ber'] for c in neanea):.3f} to {max(c['ber'] for c in neanea):.4f}), "
        f"although their word timing passes the timing line at "
        f"{sum(c['timing_distance'] <= timing['line'] for c in timed)} of the {len(timed)} timed "
        "ayat: the signature of one recording re-processed past what the fingerprint survives. "
        "A pilot recording held in Tadabur only in such a copy would be missed. The line was "
        "fixed before screening and is not retuned.",
        "",
        "**Muaalem**: the first row group of each of the 27 moshafs (the opening of al-Fatiha "
        "and al-Baqarah), each pilot ayah slid along the segments that cover it. Closest BER "
        "per mushaf over all 27: " + "; ".join(
            f"`{slug}` {min(m['closest'][slug]['ber'] for m in results['muaalem'] if slug in m['closest']):.3f}"
            for slug in PILOT) + f"; matches: "
        f"{sum(len(m['matches']) for m in results['muaalem'])}.",
        "",
    ]
    return out


def _items(report: dict) -> list[str]:
    out = ["## Items", "",
           "| mushaf | ayat | items (QUD segments) | hours | unmapped ayat | segments the "
           "phonetizer cannot handle | ayat with a repeated word |", "|---|---|---|---|---|---|---|"]
    for slug in PILOT:
        i = report["items"][slug]
        out.append(f"| `{slug}` | {i['ayat']} | {i['items']} | {i['hours']} | "
                   f"{i['unmapped_ayat']} | {i['phonetizer_unsupported']} | "
                   f"{i['repeated_words_ayat']} |")
    return out + ["", "A segment the phonetizer cannot handle ends in waqf on a leen letter "
                  "followed by its final consonant (`شَىْءٍ`, `إِلَيْهِ`, `ٱلَّيْلِ`), "
                  "quran-transcript 0.5.2's known gap; its carriers are left out.", ""]


def _directions(report: dict) -> list[str]:
    out = ["## Carriers per direction, against the mushaf (no listening)", "",
           "#144's rule reads two decodes: the base teacher on the whole segment and `h448` "
           "streamed at b = 0. **Agreed**: both commit the mushaf's letter (hold the "
           "geminate); **disagreed**: either commits something else (a pending site, a "
           "worklist row, not queued); **abstained**: neither disagrees, one commits nothing "
           "(counted only). A carrier in two pairs counts in both directions. "
           f"Counts at or above the n ≥ {FLOOR} floor are marked ✓.", "",
           "| direction | carriers | agreed | disagreed | abstained | disagreed: base "
           "another | disagreed: h448 another |", "|---|---|---|---|---|---|---|"]
    pooled = report["directions"]["pooled"]
    order = [d for family in FAMILIES.values() for d in family]
    for d in [d for d in order if d in pooled]:
        c = pooled[d]
        mark = " ✓" if c.get("agreed", 0) >= FLOOR else ""
        out.append(f"| {d} | {c['carriers']} | {c.get('agreed', 0)}{mark} | "
                   f"{c.get('disagreed', 0)} | {c.get('abstained', 0)} | "
                   f"{c.get('base/spans committed another', 0)} | "
                   f"{c.get('h448/stream_b0 committed another', 0)} |")
    out += ["", "Carriers per mushaf and class: " + "; ".join(
        f"`{slug}` " + ", ".join(f"{k} {v}" for k, v in sorted(report["classes"][slug].items()))
        for slug in PILOT) + ".", ""]
    return out


def _rates(report: dict) -> list[str]:
    out = ["## Rates per model and protocol", "",
           "Per direction, the share of carriers where each arm committed the mushaf's letter "
           "(`mushaf`), the pair's partner or a single consonant at a geminate (`swap`), or "
           "anything else, nothing included (`other`). Each mushaf is one reciter, so the "
           "intervals are clustered by surah: they describe this recording, not reciters.", ""]
    for slug in PILOT:
        rows = report["rates"][slug]["rows"]
        out += [f"### `{slug}`: swap rate", "",
                "| direction | sites | " + " | ".join(ARMS) + " |",
                "|---|---|" + "---|" * len(ARMS)]
        for row in rows:
            if not row.get("arms"):
                continue
            cells = [rate_cell(row["arms"][a][SWAP], row["counts"][a][SWAP]) for a in ARMS]
            out.append(f"| {row['direction'] if row['direction'] != 'all' else row['family'] + ' (all)'}"
                       f" | {row['sites']} | " + " | ".join(cells) + " |")
        out.append("")
        out += [f"`mushaf` and `other` for the in-word geminates in `{slug}`: " + "; ".join(
            f"{a} {rate_cell(r['arms'][a][MUSHAF], r['counts'][a][MUSHAF])} / "
            f"{rate_cell(r['arms'][a][OTHER], r['counts'][a][OTHER])}"
            for r in rows if r["direction"] == GEMINATE for a in ARMS) + ".", ""]
    return out


def _per_1000(x) -> str:
    return "–" if x is None else f"{x:.1f}"


def _app(report: dict) -> list[str]:
    app, sample, device = report["app"], report["tadabur_word_sample"], report["device_logs"]
    out = ["## The app outcome on presumably correct professional recitation", "",
           "Every word of every item, graded by Muraja's engine on the item's recorded check "
           "sequence (`checks`), per model; each session starts on the item's first canonical "
           "word and every word is looked up by its own place in the ayah. The recitation is "
           "presumed correct, so a flagged word is a presumed false flag. Beside it, per 1,000 "
           "words: the Tadabur word sample's false flags (155 correctly recited words, one per "
           "held-out reciter, weighted; `truth_baseline/report.json`) and the teacher's own "
           "device sessions (#151, `device_logs/report.json`), with the words their docs explain "
           "(skipped, re-read, never recited) left out, over all protocol eras and on the 1 s "
           "hop alone. No included device session ran `h448`.", "",
           "| model | scoring | QUD words graded | QUD flagged | QUD per 1,000 | Tadabur sample "
           "per 1,000 [interval] (flags) | device, all eras, per 1,000 (flags of words) | "
           "device, 1 s hop, per 1,000 (flags of words) |",
           "|---|---|---|---|---|---|---|---|"]

    def device_cell(model: str, group: str, scoring: str) -> str:
        if model != "base":
            return "–"
        d = device[group][scoring]["unexplained"]
        return f"{_per_1000(d['per_1000'])} ({d['flags']} of {d['words']})"

    for model in ("base", "h448"):
        for scoring in ("shipped_default", "allowances_off"):
            q, t = app[model][scoring], sample[model][scoring]
            out.append(
                f"| {model} | {scoring} | {q['graded']} | {q.get('flagged', 0)} | "
                f"{_per_1000(q['per_1000'])} | {_per_1000(1000 * t['rate'])} "
                f"[{_per_1000(1000 * t['interval'][0])}, {_per_1000(1000 * t['interval'][1])}] "
                f"({t['false_flags']} of {t['correct_words']}) | "
                f"{device_cell(model, 'teacher', scoring)} | "
                f"{device_cell(model, 'teacher, 1 s hop', scoring)} |")
    out += ["", "Diagnostic: the device sessions with every graded word, explained ones "
            "included, per 1,000: " + "; ".join(
                f"{group} {scoring} {_per_1000(device[group][scoring]['all']['per_1000'])}"
                for group in device for scoring in ("shipped_default", "allowances_off")) + "."]
    out += ["", "Causes of the QUD flags (a word may carry several; `training.word_flags`):", "",
            "| model | scoring | " + " | ".join(app["base"]["shipped_default"]["causes"]) + " |",
            "|---|---|" + "---|" * len(app["base"]["shipped_default"]["causes"])]
    for model in ("base", "h448"):
        for scoring in ("shipped_default", "allowances_off"):
            causes = app[model][scoring]["causes"]
            out.append(f"| {model} | {scoring} | " + " | ".join(str(v) for v in causes.values())
                       + " |")
    out += ["", "Where in its item a word flagged as skipped sits (the engine's skip marking; each "
            "item is one session, starting on its first word): " + "; ".join(
                f"{model} {scoring} " + ", ".join(f"{k} {v}" for k, v in
                                                  app[model][scoring]["skipped_where"].items())
                for model in ("base", "h448") for scoring in ("shipped_default",)) + "."]
    out += ["", "Per mushaf, per 1,000 words: " + "; ".join(
        f"{model} {scoring} " + ", ".join(
            f"`{slug}` {_per_1000(v['per_1000'])}" for slug, v in
            app[model][scoring]["per_mushaf"].items())
        for model in ("base", "h448") for scoring in ("shipped_default", "allowances_off")) + ".",
            ""]
    return out


def _listening(report: dict) -> list[str]:
    est = report["listening"]
    out = ["## The owner's listening, from the measured disagreements", "",
           "Each disagreed carrier's excerpt is its word and one either side (QUD's word "
           "times, #87's padding), played twice plus 4 s to answer (#87's model). A budget of "
           "*k* draws at most *k* disagreed carriers per stratum (a carrier letter, or the "
           "in-word geminates), at the stratum's mean excerpt time. No sample is drawn here.",
           "", "| scope | disagreed carriers | census | k = 20 | k = 50 |", "|---|---|---|---|---|"]
    for scope in ("pooled", *PILOT):
        e = est[scope]
        out.append(f"| {scope} | {e['disagreed_carriers']} | {e['census_minutes']} min | "
                   f"{e['k20_minutes']} min | {e['k50_minutes']} min |")
    out += ["", "Disagreed carriers per stratum (pooled): " + ", ".join(
        f"{k.removeprefix('letter:')} {v}" for k, v in est["pooled"]["per_stratum"].items()) + ".",
        ""]
    return out


def _compute(report: dict) -> list[str]:
    c = report["compute"]
    return ["## Compute and disk", "",
            f"- GPU: {c['gpu_hours']} h of wall time holding the GPU lock "
            f"({c['parallel_runs']} shard runs side by side on one {c['gpu']}), for "
            f"{c['audio_hours']} h of segment audio decoded by both models in all three "
            "protocols.",
            f"- Peak cuda-dev disk for the pilot: {c['peak_disk_gb']} GB, sampled each minute "
            f"from {c['disk_sampled_from']} (the shards being read, each deleted once read, the "
            "decodes, and the overlap check's Tadabur shards); no audio was kept.",
            f"- The overlap check's stage 2 read every Tadabur shard once (each deleted once "
            f"read), CPU only, in {c['overlap_check_wall_hours']} h of wall time over "
            f"{c['overlap_check_runs']} runs (each resumed where the last stopped), the first "
            "beside the decoding.",
            f"- Committed decodes: {c['committed_decodes_mb']} MB gzipped.", ""]


def render(report: dict) -> str:
    lines = [
        "# QUD pilot: correct-recitation eval set, measured without listening (#144)",
        "",
        "Generated by `python -m training.qud_pilot doc` from "
        "`tools/tadabur/qud_eval/pilot/report.json`. Evaluation only, never a training "
        "signal. Two professional Hafs mushafs from QUD "
        f"(revision `{report['qud_revision'][:8]}`): {', '.join(f'`{s}`' for s in PILOT)}.",
        "",
        *_overlap(),
        *_items(report),
        *_directions(report),
        *_rates(report),
        *_app(report),
        *_listening(report),
        *_compute(report),
        "## What this can and cannot say",
        "",
        "- **Can:** how often each model, under each protocol, commits a letter other than "
        "the mushaf's at a target-pair carrier or collapses an in-word geminate in two "
        "professional recordings neither model heard, and how many words the app would flag "
        "there.",
        "- **Cannot:** whether a disagreement is a model error or the reciter's: nobody has "
        "listened, so every disagreement is `pending`. The agreed sites hold only if the "
        "reciter recited correctly (`assumes_competent_reciter`), and they were chosen by "
        "both baselines committing the mushaf, so a candidate can only lose on them. Two "
        "reciters are no population: no rate here generalises to reciters, and no interval "
        "is clustered by reciter.",
        "- Nothing here sets a threshold, a margin or a floor.",
        "",
    ]
    return "\n".join(lines)

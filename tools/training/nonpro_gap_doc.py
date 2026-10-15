"""Render ``docs/nonpro-gap.md`` from #159's report (``tadabur/nonpro_gap/report.json``).

Every number is read from the report; the prose states only what the construction fixes and
what the intervals show. Nothing here sets a threshold or recommends training.
"""

from __future__ import annotations

from training.nonpro_gap import CLASSES, D2_SAMPLE_SIZE, MODELS

CLASS_NAMES = {"phoneme": "phoneme (#73)", "consonant": "consonant", "focus": "ل ن ء ب د",
               "haraka": "haraka", "shaddah": "shaddah"}
CORPORA = {
    "tadabur_dev": "Tadabur dev (all)",
    "tadabur_dev_unseen": "Tadabur dev (`h448` holds no copy)",
    "qud": "QUD (professional, external)",
    "tarteel_v1": "Tarteel v1 (app users)",
    "qdat": "qdat (amateurs)",
}
#: The D1 rows the readout compares with Tadabur, and the Tadabur row they are compared with.
EXTERNAL = ("qud", "tarteel_v1", "qdat")
HOME = "tadabur_dev_unseen"


def _pct(x: float) -> str:
    return f"{100 * x:.1f}"


def rate_cell(entry: dict | None) -> str:
    """``point [low, high]`` in percent."""
    if entry is None:
        return "–"
    return f"{_pct(entry['point'])}% [{_pct(entry['low'])}, {_pct(entry['high'])}]"


def gap_cell(entry: dict) -> str:
    """A signed difference in points with its interval."""
    def signed(x: float) -> str:
        return f"{100 * x:+.1f}"
    return f"{signed(entry['point'])} [{signed(entry['low'])}, {signed(entry['high'])}]"


def side(entry: dict) -> str:
    """Which side of zero a paired interval sits on (positive: ``h448`` moves further)."""
    if entry["low"] > 0:
        return "`h448` further (interval above zero)"
    if entry["high"] < 0:
        return "the teacher further (interval below zero)"
    return "no difference resolved (interval spans zero)"


def _d1(report: dict) -> list[str]:
    d1 = report["d1"]
    out = ["## D1: clone agreement, by corpus", "",
           "`h448`'s decode read against the base teacher's under identical windows. Cells: "
           "agreement % [95% cluster-bootstrap interval]. Clusters: reciters (Tadabur, qdat), "
           "mushaf and surah (QUD), surah (Tarteel v1).", "",
           "| corpus | protocol | items | clusters | "
           + " | ".join(CLASS_NAMES[c] for c in CLASSES) + " |",
           "|---|---|---|---|" + "---|" * len(CLASSES)]
    for corpus, name in CORPORA.items():
        for protocol, block in d1[corpus].items():
            clusters = block["rates"]["phoneme"]["clusters"]
            out.append(f"| {name} | {protocol} | {block['items']} | {clusters} | "
                       + " | ".join(rate_cell(block["rates"][c]) for c in CLASSES) + " |")
    numerics = report["numerics"]
    out += ["", "**Numerics.** Every decode above is bf16 weights at batch 1. Against the teacher "
            "decode the dev manifest caches (bf16, batch 32), this run's teacher agrees "
            f"{rate_cell(numerics['base_vs_cached_teacher']['phoneme'])} of phonemes and "
            f"`h448` {rate_cell(numerics['h448_vs_cached_teacher']['phoneme'])}; `h448`'s "
            "headline 94.87% was fp32 weights at batch 32 against the same cached decode.", ""]
    home = d1[HOME]["stream_b0"]["rates"]
    lines = []
    for corpus in EXTERNAL:
        rates = d1[corpus]["stream_b0"]["rates"]
        lines.append(f"- {CORPORA[corpus]}: " + "; ".join(
            f"{CLASS_NAMES[c]} {_pct(rates[c]['point'])}% "
            f"({100 * (rates[c]['point'] - home[c]['point']):+.1f} points)"
            for c in CLASSES))
    out += [f"**Against {CORPORA[HOME]}** at `stream_b0` (points are the external corpus "
            "minus Tadabur; the intervals above are per corpus, not of the difference):", "",
            *lines, ""]
    return out


def _grid(report: dict) -> list[str]:
    grid, sample, noise = report["grid"], report["sample"], report["noise"]
    reverb, codec = grid["reverb"], grid["codec"]
    out = ["## D2: the grid, fixed before any decode", "",
           f"Committed in `tools/tadabur/nonpro_gap/grid.json` (`{grid['version']}`, seed "
           f"{grid['seed']}) before anything was decoded; the recipe is "
           "`training.perturbation.perturb`, and every condition keeps the clip's timing "
           f"(zero lag, tested on an impulse and a chirp). **Amended:** {grid['amended']}; "
           "only the codec condition was decoded again (see the end of D2).", "",
           "| condition | kind | level |", "|---|---|---|"]
    units = {"noise": "dB SNR", "reverb": "s RT60", "band_limit": "Hz sample rate",
             "codec": "bit/s"}
    for c in grid["conditions"][1:]:
        out.append(f"| {c['name']} | {c['kind']} | {c['level']:g} {units[c['kind']]} |")
    out += ["",
            f"- **Noise:** `{grid['noise']['repo']}` at `{grid['noise']['revision'][:8]}`, shards "
            + ", ".join(f"`{s}`" for s in grid["noise"]["shards"])
            + f"; pool {noise['recordings']} recordings ({noise['hours']} h; left out: "
            + ", ".join(f"{k} {v}" for k, v in noise["left_out"].items())
            + f"). One recording and offset per clip at every SNR, looped; SNR is the clip's "
            f"mean power over the segment's. {noise['recordings_drawn']} recordings drawn, "
            "licences: " + ", ".join(f"{k} {v}" for k, v in noise["licences_drawn"].items())
            + ". The collection states no licence of its own; each recording carries its "
            "Freesound licence. Downloaded to the GPU box's scratch only; no audio is committed.",
            f"- **Reverb:** a unit direct path and a Gaussian tail from "
            f"{1000 * reverb['tail_onset_s']:g} ms decaying 60 dB over the RT60, its energy "
            f"{reverb['drr_db']:g} dB below the direct path's (DRR), one RT60 long, drawn per "
            "clip; the result scaled back to the clean clip's RMS.",
            f"- **Band limit:** resampled to {grid['band_limit']['sample_rate']} Hz and back "
            "(a 4 kHz band).",
            f"- **Codec:** Opus through ffmpeg's libopus at {codec['bitrate'] // 1000} kbit/s, "
            f"`{codec['application']}`, cutoff {codec['cutoff_hz']} Hz (narrowband): a "
            f"low-bitrate phone call; {codec['alignment']}.",
            f"- **Sample:** {D2_SAMPLE_SIZE} of the dev half's {sample['eligible']} clips "
            f"`h448` holds no recording or copy of ({len(sample['h448_seen'])} of "
            f"{sample['dev_clips']} are copies of rows in its training or validation shards), "
            f"by salted rank: {sample['reciters']} reciters, {sample['hours']} h. Each clip is "
            "decoded whole at `stream_b0` (its 5 s windows), clean and under each condition.",
            ""]
    return out


def _d2(report: dict) -> list[str]:
    d2 = report["d2"]
    conditions = [c["name"] for c in report["grid"]["conditions"][1:]]
    out = ["## D2: drift per condition", "",
           "Per class: each model's agreement with **its own** clean decode (drift), and "
           "`h448`'s with the **teacher's** clean decode (for the teacher the two are one); "
           "`h448`'s agreement with the teacher on **the same perturbed audio** (D1's metric "
           "under the condition) and its change from clean; then two paired differences, "
           "`h448`'s minus the teacher's, in points: **excess drift** (how much further `h448` "
           "moves from its own clean decode) and **excess clone drop** (how much further its "
           "agreement with the teacher's clean decode falls than the teacher's own does). "
           "Positive means `h448` moves further. Every interval is clustered by reciter, and a "
           "difference is paired (both readings of a clip resample together).", ""]
    for c in CLASSES:
        clean = d2["clean"]["teacher"]["h448"][c]
        out += [f"### {CLASS_NAMES[c]}", "",
                f"Clean, `h448` against the teacher: {rate_cell(clean)}.", "",
                "| condition | base vs own clean | h448 vs own clean | excess drift | "
                "h448 vs teacher clean | excess clone drop | h448 vs teacher, same audio | "
                "its change from clean |", "|---|---|---|---|---|---|---|---|"]
        for name in conditions:
            cell = d2[name]
            out.append(f"| {name} | {rate_cell(cell['own']['base'][c])} | "
                       f"{rate_cell(cell['own']['h448'][c])} | "
                       f"{gap_cell(cell['excess_drift'][c])} | "
                       f"{rate_cell(cell['teacher']['h448'][c])} | "
                       f"{gap_cell(cell['excess_clone_drop'][c])} | "
                       f"{rate_cell(cell['clone'][c])} | {gap_cell(cell['clone_change'][c])} |")
        out.append("")
    return out + _amendment(d2)


def _amendment(d2: dict) -> list[str]:
    """What the codec row was before grid v2 aligned the codec's output with its input."""
    before, after = d2["superseded"]["opus_12k_nb_v1"], d2["opus_12k_nb"]

    def cells(cell: dict, c: str) -> str:
        return (f"base {_pct(cell['own']['base'][c]['point'])}%, h448 "
                f"{_pct(cell['own']['h448'][c]['point'])}%, excess drift "
                f"{gap_cell(cell['excess_drift'][c])}, same-audio clone "
                f"{_pct(cell['clone'][c]['point'])}%")

    return ["### The codec row under grid v1", "",
            "Grid v1's codec output lagged its input by 1-2 samples. Its decodes are kept "
            "(`decodes/superseded/`), read the same way; v1 → the current grid:", "",
            *(f"- {CLASS_NAMES[c]}: {cells(before, c)} → {cells(after, c)}." for c in CLASSES),
            ""]


def _readout(report: dict) -> list[str]:
    d1, d2 = report["d1"], report["d2"]
    conditions = [c["name"] for c in report["grid"]["conditions"][1:]]
    home = d1[HOME]["stream_b0"]["rates"]
    out = ["## What the evidence says", "",
           "Computed from the tables above; the grid's conditions are synthetic stand-ins, not "
           "measured properties of Tarteel v1 or qdat, so a match in level is not a match in "
           "cause.", ""]
    for c in ("phoneme", "consonant", "focus", "haraka", "shaddah"):
        external = {k: d1[k]["stream_b0"]["rates"][c]["point"] for k in EXTERNAL}
        clone = {n: d2[n]["clone"][c]["point"] for n in conditions}
        reached = {k: [n for n in conditions if clone[n] <= v] for k, v in external.items()
                   if k != "qud"}
        out.append(
            f"- **{CLASS_NAMES[c]}.** D1, `h448` against the teacher at `stream_b0`: "
            f"{_pct(home[c]['point'])}% on {CORPORA[HOME]}; "
            + ", ".join(f"{CORPORA[k]} {_pct(v)}% ({100 * (v - home[c]['point']):+.1f})"
                        for k, v in external.items())
            + f". D2, the same agreement on the same perturbed audio: clean "
            f"{_pct(d2['clean']['teacher']['h448'][c]['point'])}%, "
            + ", ".join(f"{n} {_pct(v)}%" for n, v in clone.items())
            + ". At or below " + "; at or below ".join(
                f"{CORPORA[k]}'s level: " + (", ".join(v) or "no condition")
                for k, v in reached.items())
            + ". Excess drift: " + "; ".join(f"{n} {side(d2[n]['excess_drift'][c])}"
                                             for n in conditions) + ".")
    return out + [""]


def _compute(report: dict) -> list[str]:
    c = report["compute"]
    return ["## Compute and disk", "",
            f"- GPU: {c['gpu_hours']} h of wall time holding the GPU lock ({c['parallel_runs']} "
            f"condition runs side by side on one {c['gpu']}), for {c['audio_hours']} h of audio "
            "of committed decodes per model at `stream_b0` (the superseded codec decodes besides)."
            + (f" {c['run_notes']}" if c.get("run_notes") else ""),
            f"- Peak scratch disk on cuda-dev: {c['peak_scratch_gb']} GB sampled each minute "
            f"from {c['disk_sampled_from']} (the noise pool and the decodes; the two noise "
            "shards, about 1 GB, were each deleted once read, between samples).", ""]


def render(report: dict) -> str:
    models = " and ".join(f"`{m}`" for m in MODELS)
    head = ["# Diagnosing `h448`'s non-professional gap: clone agreement and acoustic drift "
            "(#159)", "",
            "Generated by `python -m training.nonpro_gap doc` from "
            "`tools/tadabur/nonpro_gap/report.json`. Evaluation only; nothing here trains, sets "
            "a threshold or recommends one. Never used: the `decode_evalset` test half, the "
            "sealed panel, Ship A's held-out half.", "",
            f"#156 found `h448` 2-3x worse than the base teacher on non-professional audio. "
            f"Two diagnostics, both reading {models}'s decodes phoneme to phoneme (#73's "
            "metric, never through a scorer), separate **domain shift** (D1: does `h448` stop "
            "cloning the teacher on audio from outside Tadabur?) from **acoustic fragility** "
            "(D2: does `h448` drift further than the teacher when the same Tadabur audio is "
            "noisy, reverberant or phone-band?). Per class, through one Levenshtein alignment "
            "of the decode to the reference decode (`training.nonpro_gap.tally`): every phoneme "
            "(pooled 1 − edits / reference phonemes), every consonant run (agreeing when a "
            "letter of it is matched), the group ل ن ء ب د, every fatha, damma and kasra, every "
            "geminate's shaddah (agreeing when its whole run is matched). A slot the other "
            "decode leaves out is a disagreement, and a decode agrees with itself in every "
            "class. All decodes are bf16 weights at batch 1 (the truth baseline's protocol).",
            ""]
    return "\n".join(head + _d1(report) + _grid(report) + _d2(report) + _readout(report)
                     + _compute(report)).rstrip() + "\n"

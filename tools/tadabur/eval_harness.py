"""Write the fixture-side, decode-level report (#55) from #84's decode caches.

The pure core is :mod:`tadabur.eval_report`; this is its runner. It decodes nothing of its own:

1. **Sites.** The re-located P3.5 fixtures (``truth_sites/p35_fixtures.jsonl``), with any
   site-level verdict from the listening session applied by site id
   (:func:`tadabur.listening_session.adjudicated`), and each site's fixture side from the
   re-location record (``p35_fixtures.relocation.jsonl``).
2. **Decodes.** The base teacher and ``h448``, each whole-span and streamed at b = 0, read from
   #84's caches through :func:`training.truth_baseline.update_decodes`, which checks the
   model, its weights' identity and every decode fingerprint. An item the cache lacks is
   decoded there, by #84's decode path, only when ``--audio-dir`` is given.
3. **Report.** :func:`tadabur.eval_report.fixture_report`, written as :data:`REPORT_PATH`, and
   the human-readable :data:`DOC_PATH` rendered from it.

Re-running after the listening session (#61) fills the real-mistake side in with no other
change. A candidate is added with ``--model NAME=REF`` (its decodes cache under ``NAME``, as
in ``training.truth_baseline``).

Usage (from ``tools/``)::

    # Anywhere, torch-free, from the committed decodes:
    python -m tadabur.eval_harness
    # On the GPU box, decoding what a cache lacks (e.g. a candidate):
    python -m tadabur.eval_harness --audio-dir /root/scratch/issue-83/stage/clips \\
        --model base=obadx/muaalem-model-v3_2 --model cand=<checkpoint> --out <json> --doc <md>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.tashkeel_eval import write_text_atomically
from training.truth_baseline import (
    BASELINE_DIR,
    DEFAULT_MODELS,
    PROTOCOLS,
    arm,
    file_sha256,
    items_of,
    p35_reproduction,
    update_decodes,
)
from training.truth_baseline_doc import NOT_CERTIFIABLE, SPARSE_MARK, format_estimate
from training.site_outcomes import CORRECT_SIDE, MISTAKE_SIDE

from .eval_fixtures import ACCEPT, REJECT
from .eval_report import (
    AS_HEARD,
    AS_PARTNER,
    NO_COMMIT,
    NO_SITE,
    OTHER_CONSONANT,
    SHOULD_ACCEPT,
    SHOULD_REJECT,
    SUFFICIENT,
    TOO_SMALL,
    fixture_report,
)
from .listening_session import VERDICTS_PATH, adjudicated, read_verdicts
from .p35_truth_sites import RELOCATION_PATH, SITES_PATH
from .staged_audio import REGISTRY_PATH, load_staged_clips
from .truth_sites import SHADDAH, load_truth_sites

REPORT_PATH = BASELINE_DIR / "fixture_sides.json"
DOC_PATH = Path(__file__).parent.parent.parent / "docs" / "fixture-sides.md"

_FIXTURE_SIDE_OF_VERDICT = {ACCEPT: SHOULD_ACCEPT, REJECT: SHOULD_REJECT}


def fixture_sides(relocation_path: Path = RELOCATION_PATH) -> dict[str, str]:
    """Each re-located site's fixture side, from the verdict of the fixture it came from."""
    sides = {}
    for line in relocation_path.read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        for site_id in row["site_ids"]:
            sides[site_id] = _FIXTURE_SIDE_OF_VERDICT[row["verdict"]]
    return sides


def build(models: dict[str, str], audio_dir: Path | None) -> dict:
    """Load the sites, verdicts and decode caches and score them (module docstring)."""
    sites = adjudicated(load_truth_sites(SITES_PATH), read_verdicts(VERDICTS_PATH))
    registry = load_staged_clips(REGISTRY_PATH)
    items = items_of(sites, registry)
    decodes: dict[str, dict[str, str]] = {}
    arm_models: dict[str, dict] = {}
    for name, ref in models.items():
        cache = update_decodes(name, ref, items, registry, audio_dir)
        for protocol in PROTOCOLS:
            decodes[arm(name, protocol)] = {key: cache["items"][key][protocol] for key in items}
            arm_models[arm(name, protocol)] = {
                "model": name,
                "protocol": protocol,
                "identity": cache["identity"],
                "decode": cache["fingerprints"][protocol],
            }
    reciter_of = {name: clip.reciter_id for name, clip in registry.items()}
    report = fixture_report(sites, fixture_sides(), reciter_of, decodes, arm_models)
    report["inputs"] = {
        path.name: file_sha256(path) for path in (SITES_PATH, RELOCATION_PATH, VERDICTS_PATH, REGISTRY_PATH)
    }
    if "base" in models:
        report["p35_base_reproduction"] = p35_reproduction(sites, decodes[arm("base", "spans")])
    return report


# --- the rendered report ---------------------------------------------------------------

_SIDE_TITLES = {
    CORRECT_SIDE: "Correct recitation (heard = prescribed)",
    MISTAKE_SIDE: "Real mistakes (heard ≠ prescribed)",
}
_ROLE_HEADINGS = {
    CORRECT_SIDE: {
        AS_HEARD: "as heard (faithful)",
        AS_PARTNER: "partner (misheard)",
        OTHER_CONSONANT: "other letter (misheard)",
        NO_COMMIT: "no commit",
    },
    MISTAKE_SIDE: {
        AS_HEARD: "as heard (mistake heard)",
        AS_PARTNER: "mushaf's value (collapsed)",
        OTHER_CONSONANT: "other letter",
        NO_COMMIT: "no commit (missed)",
    },
}
_ROLES = (AS_HEARD, AS_PARTNER, OTHER_CONSONANT, NO_COMMIT)
_SUPPORT = {SUFFICIENT: "sufficient", TOO_SMALL: SPARSE_MARK, NO_SITE: "no site"}


def _family_name(family: str) -> str:
    return f"{family} (provisional)" if family == SHADDAH else family


def _size(row: dict) -> str:
    size = f"{row['sites']} / {row['reciters']}"
    return f"{size} ({SPARSE_MARK})" if row["support"] == TOO_SMALL else size


def _role_cell(row: dict, arm_: str, role_: str) -> str:
    if not row["sites"]:
        return "–"
    entry = row["arms"][arm_].get(role_)
    if entry is None:  # a shaddah row has no third letter
        return ""
    return f"{entry['sites']} · {format_estimate(entry)}"


def _side_table(rows: list[dict], side: str, arm_: str) -> list[str]:
    headings = _ROLE_HEADINGS[side]
    lines = [
        "| family | prescribed → heard | sites / reciters | fixtures (accept / reject) | "
        + " | ".join(headings[r] for r in _ROLES) + " |",
        "|---|---|---|---|" + "---|" * len(_ROLES),
    ]
    for row in rows:
        origin = f"{row['fixture_sides'][SHOULD_ACCEPT]} / {row['fixture_sides'][SHOULD_REJECT]}"
        lines.append(
            f"| {_family_name(row['family'])} | {row['prescribed']} → {row['heard']} | {_size(row)} "
            f"| {origin} | " + " | ".join(_role_cell(row, arm_, r) for r in _ROLES) + " |"
        )
    return lines


def _pending_table(pending: list[dict]) -> list[str]:
    lines = ["| family | prescribed | heard | fixture side | sites |", "|---|---|---|---|---|"]
    for p in pending:
        lines.append(
            f"| {_family_name(p['family'])} | {p['prescribed']} | `{p['heard']}` "
            f"| {p['fixture_side']} | {p['sites']} |"
        )
    return lines


def _side_section(report: dict, side: str) -> list[str]:
    conventions = report["sign_conventions"][side]
    lines = [f"## {_SIDE_TITLES[side]}", "", f"{conventions['side'].capitalize()}. Roles:", ""]
    lines += [f"- **{_ROLE_HEADINGS[side][r]}**: {conventions[r]}." for r in _ROLES]
    lines.append("")
    rows = [r for r in report["rows"] if r["side"] == side]
    if not any(r["sites"] for r in rows):
        waiting = sum(p["sites"] for p in report["pending"] if p["fixture_side"] == SHOULD_REJECT)
        lines += [
            f"**Pending adjudication.** No site has a real-mistake verdict yet, so this side has no "
            f"numbers. The {waiting} sites of the should-reject fixtures wait for their site-level "
            "re-listen (#61); `python -m tadabur.eval_harness` fills this side in once their "
            "verdicts are in `tadabur/listening_session/verdicts.jsonl`.",
            "",
        ]
        return lines
    for arm_ in report["arms"]:
        lines += [f"### {arm_}", "", "Sites committed in each role · % of the row [95% interval]", ""]
        lines += _side_table(rows, side, arm_) + [""]
    return lines


def render(report: dict) -> str:
    """The Markdown report: every number comes from ``report``."""
    fp = report["fingerprints"]
    reproduction = report.get("p35_base_reproduction")
    lines = [
        "# Fixture sides: the decode on the P3.5 fixtures, never pooled (#55)",
        "",
        "Generated by `python -m tadabur.eval_harness` (from `tools/`); do not edit by hand. The",
        "pure core is `tools/tadabur/eval_report.py`; the machine-readable report, with every",
        "site's outcome per arm for the paired diff (#57), is",
        "`tools/tadabur/truth_baseline/fixture_sides.json`. Each site's outcome is #84's",
        "(`training.site_outcomes`), read from #84's decode caches; nothing is decoded again.",
        "",
        "## Read this first",
        "",
        "- **Two sides, never pooled** (ADR-0008, acceptance rules §1). One confusion cell means",
        "  opposite things on the two sides: committing the partner letter is a mishearing where",
        "  the reciter said the mushaf's letter, and a collapse onto the reference where the",
        "  reciter said the partner. Each side has its own table and its own role names.",
        "- **Fixture side → recitation side.** A should-accept fixture says the mushaf's value was",
        "  said at its site (correct recitation). A should-reject fixture is a clip-level verdict",
        "  and never says what was said at a site (§8), so its sites are `pending` until the",
        "  listening session hears each one; a site then joins the side its verdict gives it. The",
        "  *fixtures* column of each row counts where its sites came from.",
        "- **Rows are directional sites, not aligned columns.** One row per pair direction",
        "  (`ذ↔ظ` included, §7) and gemination state, read at the re-located truth sites (#83).",
        "- **The P3.5 sites were selected on the base teacher's errors** (#83 kept a site only where",
        "  the base whole-span decode showed the contrast), so on the correct side `base/spans`",
        "  commits the partner by selection, not by measurement."
        + (
            f" Its decode here reproduces the re-location decode on {reproduction['identical']} of "
            f"{reproduction['items']} items."
            if reproduction else ""
        ),
        "- **Intervals** follow §1: the reciter-clustered bootstrap (B = "
        f"{report['bootstrap']['replicates']:,}, seed {report['bootstrap']['seed']}). Every clip",
        "  has one reciter, so reciter clusters nest clips (ADR-0008 asks for clip-clustered",
        "  intervals; this is at least as conservative). **W** marks a Wilson bound on a sparse or",
        f"  degenerate row whose sites are independent and equally weighted; **({NOT_CERTIFIABLE})**",
        "  marks a value with no admissible bound.",
        f"- **{SPARSE_MARK}**: a row with fewer than 10 reciters or 20 sites cannot support a claim.",
        "  Nothing here is a verdict; a family without sufficient support on both sides is",
        "  \"in scope, insufficient evidence\" (§7).",
        "- **Shaddah is provisional** until #92 freezes its state table.",
        "- **`strict_accept` is not here.** It is ADR-0001's training-data hygiene gate",
        "  (`tadabur.scorer.strict_accept`), not a measure of the fine-tune.",
        "",
        "## Support per family (§7)",
        "",
        "The best direction's support on each side, counting sites with a verdict only.",
        "",
        "| family | correct recitation | real mistakes | sites without a verdict | status |",
        "|---|---|---|---|---|",
    ]
    for f in report["families"]:
        lines.append(
            f"| {_family_name(f['family'])} | {_SUPPORT[f[CORRECT_SIDE]]} | "
            f"{_SUPPORT[f[MISTAKE_SIDE]]} | {f['without_verdict']} | {f['status']} |"
        )
    lines.append("")
    for side in (CORRECT_SIDE, MISTAKE_SIDE):
        lines += _side_section(report, side)
    lines += [
        "## Sites without a verdict",
        "",
        "`pending` (not adjudicated at site level yet) and `unclear` sites leave every row.",
        "",
    ]
    lines += (_pending_table(report["pending"]) if report["pending"] else ["None."]) + [""]
    lines += [
        "## Fingerprints",
        "",
        "A paired diff (#57) refuses two reports whose fixture or schema fingerprints differ.",
        "",
        "| what | value |",
        "|---|---|",
        f"| fixtures (scored sites, fixture sides, reciters) | `{fp['fixtures']}` |",
        f"| schema | `{fp['schema']}` |",
    ]
    lines += [f"| input `{name}` | `{sha}` |" for name, sha in sorted(report.get("inputs", {}).items())]
    lines += ["", "| arm | weights | decode mode | decodes sha256 |", "|---|---|---|---|"]
    for arm_ in report["arms"]:
        m = fp["arms"][arm_]
        identity = m["identity"]
        weights = identity.get("checkpoint_sha256") or identity.get("hub_revision")
        lines.append(
            f"| {arm_} | `{identity['model_ref']}` @ `{weights}` | `{m['decode']['mode']}`, "
            f"{m['decode']['weights_dtype']}, batch {m['decode']['batch_size']} | `{m['decodes_sha256']}` |"
        )
    lines += [
        "",
        "## Re-running",
        "",
        "```bash",
        "# from tools/, torch-free, from the committed decode caches",
        "python -m tadabur.eval_harness",
        "```",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-dir", type=Path, help="staged 16 kHz clips; needed only to decode")
    parser.add_argument(
        "--model", action="append", default=[], metavar="NAME=REF",
        help="a model to score (default: base and h448); its decodes cache under NAME",
    )
    parser.add_argument("--out", type=Path, default=REPORT_PATH, help="the JSON report")
    parser.add_argument("--doc", type=Path, default=DOC_PATH, help="the rendered Markdown report")
    args = parser.parse_args()
    models = dict(m.split("=", 1) for m in args.model) or DEFAULT_MODELS

    report = build(models, args.audio_dir)
    write_text_atomically(args.out, json.dumps(report, ensure_ascii=False, indent=1) + "\n")
    write_text_atomically(args.doc, render(report))
    print(f"Wrote {args.out} and {args.doc}")


if __name__ == "__main__":
    main()

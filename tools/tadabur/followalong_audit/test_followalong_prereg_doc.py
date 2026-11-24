"""Acceptance tests for the follow-along audit's pre-registration documents (#310).

Committed before the documents exist and **locked**: later turns may add tests but never weaken
or rewrite these. The change is doc-only, so the tests guard the facts the documents pin, not
their prose: every identity, checksum, count and pool a document quotes is recomputed from the
committed S0-S3 artifacts, and the frozen documents it builds on stay byte-identical.

Contract this file pins
-----------------------
* ``docs/followalong-audit-preregistration.md`` (new, dated, frozen before S5's draw) has nine
  numbered ``## N. Title`` sections in the order of issue #310: Population, Arms, Strata,
  Verdicts, Attribution, Decision rule, Estimates, Pilot, Dataset.
* ``docs/tlog-followalong-amendment.md`` (new, dated) records the reuse of confirm.1/2 under
  ``docs/tlog-mining-amendment.md`` section 1 and cites the frozen holdout document's SHA-256,
  the mining amendment's, and the four audit commits (S0 c4bdea30, S1 c7c6cd07, S3 9f3f6da7,
  S2 314dd865) as full 40-hex SHAs. The frozen holdout document is never edited.
* Counts are quoted with thousands separators (``5,156``); checksums as full lowercase 64-hex
  SHA-256 strings.
* Section 3's S1 table has one row per ``cause | base flags (yes/no) | words | n`` cell (14 rows,
  causes by *primary* cause, D2); its stratum table has rows starting ``S1``, ``S3``, ``S4``,
  ``S5`` and ``repeats`` with cells ``pool | n | pilot | inclusion`` (inclusion to 3 significant
  figures: ``f"{n / pool:.3g}"``).
* The approval lines read ``Owner approval: pending`` and ``Principal cross-check: pending``, or
  carry an issue-310 comment URL (D14).
"""

import gzip
import json
import re
import subprocess
from hashlib import sha256
from pathlib import Path

import pytest

from tadabur.followalong_audit.allocation import allocate

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
AUDIT = ROOT / "tools" / "tadabur" / "followalong_audit"
PREREG = DOCS / "followalong-audit-preregistration.md"
AMENDMENT = DOCS / "tlog-followalong-amendment.md"
FROZEN = DOCS / "tlog-holdout-preregistration.md"
MINING_AMENDMENT = DOCS / "tlog-mining-amendment.md"
READOUTS = ROOT / "tools" / "tadabur"

FROZEN_SHA256 = "204fe7f980488aff73402c56f1729f7f17f7c9471ea0664f6365bb8870b62086"
FIRST_DATE = "2026-11-24"
COMMITS = {"S0": "c4bdea30", "S1": "c7c6cd07", "S3": "9f3f6da7", "S2": "314dd865"}
STUDENT_PT_SHA256 = "1193e2117b56381ccc078a5b5e21d61d0ad0903565dcca0cb110b39ce24ed4a9"
MODELS_ZIP_SHA256 = "f8dc3311fe4506c505b22c4a205d30fdcf8c3b7a45b9cd1ae9008d1f7b12c439"
PRIMARY_ORDER = ("target_pair", "soft_pair", "other_consonant", "shaddah", "tashkeel", "skipped",
                 "unrecorded")
SECTIONS = ("Population", "Arms", "Strata", "Verdicts", "Attribution", "Decision rule",
            "Estimates", "Pilot", "Dataset")
STRATUM_SIZES = {"S1": 200, "S3": 250, "S4": 60, "S5": 30}
PILOT = {"S1": 56, "S3": 69, "S4": 17, "S5": 8}
BUCKETS = ("F1", "F1-align", "F-ref", "F2", "F3", "M1", "M1r", "M1u", "M2", "M3")
APPROVAL = r"(pending|https://github\.com/sysofwan/HifzGuide/issues/310#issuecomment-\d+)"


def _json(relative: str) -> dict:
    return json.loads((AUDIT / relative).read_text(encoding="utf-8"))


def _rows(relative: str) -> list[dict]:
    with gzip.open(AUDIT / relative, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


SUMMARY = _json("app_arm/summary.json")
POOLS = _json("app_arm/pools.json")
STRATA = POOLS["pools"]["strata"]


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _number(value: int) -> str:
    return f"{value:,}"


def _text(path: Path) -> str:
    assert path.is_file(), f"{path.relative_to(ROOT)} does not exist"
    return path.read_text(encoding="utf-8")


def _sections(text: str) -> dict[str, str]:
    """Each numbered ``## N. Title`` section's text, keyed by its number."""
    parts = re.split(r"^## (\d+)\. .*$", text, flags=re.M)
    return {number: body for number, body in zip(parts[1::2], parts[2::2])}


def _titles(text: str) -> list[tuple[str, str]]:
    return re.findall(r"^## (\d+)\. (.*)$", text, flags=re.M)


def _section(number: int) -> str:
    return _sections(_text(PREREG))[str(number)]


def _expect(text: str, *patterns: str) -> None:
    for pattern in patterns:
        assert re.search(pattern, text, flags=re.I | re.S), f"missing /{pattern}/"


def _cells(line: str) -> list[str]:
    return [cell.strip().strip("`*") for cell in line.strip().strip("|").split("|")]


def _int(cell: str) -> int:
    return int(cell.replace(",", ""))


def _table_rows(section: str) -> list[list[str]]:
    return [_cells(line) for line in section.splitlines() if line.strip().startswith("|")]


def _full_sha(short: str) -> str:
    return subprocess.run(["git", "rev-parse", short], cwd=ROOT, check=True, text=True,
                          capture_output=True).stdout.strip()


# --- the S1 primary-cause cells, recomputed ---------------------------------------------------------
def _s1_primary_totals() -> dict[str, int]:
    """App strict flags the oracle does not flag, by primary cause, from the committed tables."""
    oracle = {(row["item"], row["word"]): row["strict"]
              for row in _rows("oracle_sweep/word_grades.jsonl.gz")}
    totals = dict.fromkeys(PRIMARY_ORDER, 0)
    for row in _rows("app_arm/word_grades.jsonl.gz"):
        cell = row["strict"]
        if not (cell["graded"] and cell["flagged"]):
            continue
        shared = oracle.get((row["item"], row["word"]))
        if shared and shared["graded"] and shared["flagged"]:
            continue
        totals[next(cause for cause in PRIMARY_ORDER if cause in cell["causes"])] += 1
    return totals


def _s5_split() -> dict[str, int]:
    """S5's words by where they sit in the clip, and the clips the App left wholly ungraded.

    A word is *leading* before the clip's first graded word (every word of a wholly ungraded
    clip is leading), *trailing* after its last, *interior* between.
    """
    by_clip: dict[str, list[tuple[int, bool]]] = {}
    for row in _rows("app_arm/word_grades.jsonl.gz"):
        by_clip.setdefault(row["item"], []).append((row["word"], row["strict"]["graded"]))
    split = {"leading": 0, "trailing": 0, "interior": 0, "ungraded clips": 0}
    for words in by_clip.values():
        graded = [word for word, ok in sorted(words) if ok]
        if not graded:
            split["ungraded clips"] += 1
        for word, ok in words:
            if ok:
                continue
            if not graded or word < graded[0]:
                split["leading"] += 1
            elif word > graded[-1]:
                split["trailing"] += 1
            else:
                split["interior"] += 1
    return split


# --- the documents are new, dated, and build on frozen ones -----------------------------------------
def test_the_frozen_holdout_document_is_unchanged():
    assert _sha256(FROZEN) == FROZEN_SHA256


@pytest.mark.parametrize("path", [PREREG, AMENDMENT])
def test_each_document_is_dated_on_or_after_the_audit_s_last_commit_and_frozen(path):
    text = _text(path)
    status = re.search(r"\*\*Status:\*\*[^\n]*(?:\n[^\n]+)*", text)
    assert status, "no **Status:** paragraph"
    dates = re.findall(r"\b(20\d\d-\d\d-\d\d)\b", status.group(0))
    assert dates and dates[0] >= FIRST_DATE
    _expect(status.group(0), r"frozen", r"before (any|the) [\w ]{0,30}(draw|listening)")


def test_the_amendment_records_the_reuse_under_the_mining_amendment_and_claims_nothing_new():
    text = _text(AMENDMENT)
    _expect(text, r"tlog-mining-amendment\.md", r"§\s*1|section 1", r"no new claim",
            r"already mined", r"never sealed again|not sealed again", r"confirm\.1",
            r"confirm\.2", r"#310")


def test_the_amendment_cites_the_frozen_documents_by_sha256():
    text = _text(AMENDMENT)
    assert FROZEN_SHA256 in text
    assert _sha256(MINING_AMENDMENT) in text
    _expect(text, r"never edited|byte-identical|stays? unchanged")


def test_the_amendment_and_the_preregistration_cite_the_four_audit_commits_in_full():
    for path in (AMENDMENT, PREREG):
        text = _text(path)
        for short in COMMITS.values():
            assert _full_sha(short) in text, f"{path.name} does not cite {short} in full"


def test_both_documents_carry_the_two_approval_lines_pending_or_recorded():
    for path in (AMENDMENT, PREREG):
        text = _text(path)
        for line in ("Owner approval", "Principal cross-check"):
            assert re.search(rf"^\W*{line}:\W*{APPROVAL}\W*$", text, flags=re.M), \
                f"{path.name}: no '{line}: pending' or issue-310 comment line"


# --- the nine sections ------------------------------------------------------------------------------
def test_the_preregistration_has_the_nine_sections_of_the_issue_in_order():
    titles = _titles(_text(PREREG))
    assert [number for number, _ in titles[:9]] == [str(n) for n in range(1, 10)]
    for (_, title), keyword in zip(titles, SECTIONS):
        assert keyword.lower() in title.lower(), f"{title!r} is not the {keyword} section"


def test_section_1_names_the_population_and_discloses_its_reuse():
    text = _section(1)
    _expect(text, r"placeable phone audio", r"5,742", r"5,762", r"#248",
            r"warm-start[^.]*round 2|round 3[^.]*warm-start",
            r"EMA", r"item 7", r"150-word catch-loss", r"#261", r"gating[^.]*teacher",
            r"tlog-followalong-amendment\.md")
    assert SUMMARY["population"]["clips"] == 5742


def test_section_2_pins_the_app_arm_identity_by_checksum_and_commit():
    text = _section(2)
    run = SUMMARY["run"]
    for value in (MODELS_ZIP_SHA256, STUDENT_PT_SHA256, run["muraja_commit"],
                  run["harness"]["binary_sha256"], run["driver_sha256"],
                  run["audio_manifest_sha256"], run["word_grades_sha256"]):
        assert value in text, f"section 2 does not quote {value}"
    assert run["release"] == "v1.0.7" and run["latency_ms"] == 40
    _expect(text, r"v1\.0\.7", r"028bc51", r"40 ms", r"Student-PT", r"Teacher", r"Oracle")
    teacher = next(line for line in text.splitlines() if re.search(r"\bTeacher\b", line)
                   and re.search(r"[0-9a-f]{64}", line))
    assert teacher


def test_section_2_reports_every_gate_with_its_result_and_commit():
    text = _section(2)
    p2, p3 = SUMMARY["p2"], SUMMARY["p3"]
    _expect(text, r"\bP1\b[^\n]*\b74\b", r"\bP1b\b", r"\bP2\b", r"\bP3\b")
    for number in (p2["subset"]["agree"], p2["subset"]["words"], p2["all"]["agree"],
                   p2["all"]["words"]):
        assert _number(number) in text
    _expect(text, r"98\.5\s?%", r"98\.0\s?%", r"97\s?%")
    for scoring in ("strict", "shipped_default"):
        share = p3["cadence"][scoring]["share"]
        assert f"{share:.1%}" in text
    _expect(text, r"2\s?%")
    oracle = SUMMARY["oracle"]
    assert (oracle["shared"], oracle["app_flagged"]) == (1, 5157)
    _expect(text, r"1\s*(/|of)\s*5,157", r"0\.019\s?%", r"no (oracle )?pattern|0 oracle candidates")


def test_the_gates_cite_the_commits_that_produced_them():
    text = _section(2)
    for short in COMMITS.values():
        assert _full_sha(short) in text or short in text, f"section 2 never cites {short}"


def test_the_caveats_of_s3_and_s2_are_disclosed():
    text = _text(PREREG)
    _expect(text, r"payload_probe\.json", r"E5RT", r"#248", r"(8|eight)[^.\n]*unreadable|unreadable[^.\n]*(8|eight)",
            r"diacritic")


# --- strata, pools, cells and the pilot -------------------------------------------------------------
def test_section_3_quotes_every_pool_count_of_the_summary():
    text = _section(3)
    for stratum, words in STRATA.items():
        assert _number(words) in text, f"section 3 does not quote {stratum} = {words:,}"
    assert STRATA["S1"] + STRATA["S1-pending"] == 5156
    assert _number(sum(STRATA.values())) in text
    _expect(text, r"not sampled", r"residual", r"oracle-explained")


def test_section_3_sizes_the_strata_and_keeps_s4_as_the_control():
    rows = {cells[0].split(":")[0].split()[0]: cells for cells in _table_rows(_section(3))
            if len(cells) >= 5 and re.match(r"(S[1345]|repeats)\b", cells[0])}
    assert set(rows) == {*STRATUM_SIZES, "repeats"}
    pools = {"S1": STRATA["S1"] + STRATA["S1-pending"], "S3": STRATA["S3"],
             "S4": STRATA["S4"], "S5": STRATA["S5"]}
    for stratum, size in STRATUM_SIZES.items():
        pool, n, pilot, inclusion = rows[stratum][1:5]
        assert (_int(pool), _int(n), _int(pilot)) == (pools[stratum], size, PILOT[stratum])
        assert inclusion == f"{size / pools[stratum]:.3g}"
    assert (_int(rows["repeats"][2]), _int(rows["repeats"][3])) == (25, 7)
    text = _section(3)
    _expect(text, r"\b540\b", r"\b150\b", r"S4[^.\n]*(kept|control)")


def test_the_pilot_is_the_largest_remainder_split_of_the_540_stratum_snips():
    assert allocate(STRATUM_SIZES, 150) == PILOT
    assert sum(STRATUM_SIZES.values()) == 540


def test_section_3_states_the_primary_cause_order_and_the_one_cell_per_word_rule():
    text = _section(3)
    assert ", ".join(PRIMARY_ORDER) in text
    _expect(text, r"primary cause", r"floor (of )?3", r"largest remainder",
            r"soft_pair[^.\n]*target_pair|target_pair[^.\n]*soft_pair")


def test_section_3_s1_table_has_the_14_primary_cause_cells_that_sum_to_the_pool():
    cells = [row for row in _table_rows(_section(3))
             if len(row) >= 4 and row[0] in PRIMARY_ORDER and row[1] in ("yes", "no")]
    assert len(cells) == 14
    assert {(row[0], row[1]) for row in cells} == {(c, b) for c in PRIMARY_ORDER
                                                   for b in ("yes", "no")}
    counts = {(row[0], row[1] == "yes"): _int(row[2]) for row in cells}
    assert sum(counts.values()) == 5156
    assert sum(_int(row[3]) for row in cells) == STRATUM_SIZES["S1"]


def test_the_s1_cells_agree_with_the_committed_word_tables_and_pools():
    cells = [row for row in _table_rows(_section(3))
             if len(row) >= 4 and row[0] in PRIMARY_ORDER and row[1] in ("yes", "no")]
    counts = {(row[0], row[1] == "yes"): _int(row[2]) for row in cells}
    totals = _s1_primary_totals()
    assert sum(totals.values()) == 5156
    for cause in PRIMARY_ORDER:
        assert counts[(cause, True)] + counts[(cause, False)] == totals[cause], cause
    any_cause = {(row["cause"], row["base_flags"]): row["words"]
                 for row in POOLS["s1_by_cause_and_base"]}
    for cell, words in counts.items():
        assert words <= any_cause[cell], f"{cell} holds more words than all its causes' pool"


def test_the_s1_allocation_is_the_pre_registered_floor_and_largest_remainder_split():
    cells = [row for row in _table_rows(_section(3))
             if len(row) >= 4 and row[0] in PRIMARY_ORDER and row[1] in ("yes", "no")]
    counts = {(row[0], row[1] == "yes"): _int(row[2]) for row in cells}
    quoted = {(row[0], row[1] == "yes"): _int(row[3]) for row in cells}
    assert quoted == allocate(counts, 200, floor=3)


def test_section_3_splits_s5_by_where_the_word_sits_in_the_clip():
    text = _section(3)
    split = _s5_split()
    assert sum(split[kind] for kind in ("leading", "trailing", "interior")) == STRATA["S5"]
    assert split == {"leading": 3933, "trailing": 5103, "interior": 306, "ungraded clips": 250}
    for value in split.values():
        assert _number(value) in text
    _expect(text, r"leading[^.\n]*10", r"trailing[^.\n]*10", r"interior[^.\n]*10",
            r"not_in_clip")


def test_section_3_discloses_that_s4_means_no_detector_that_graded_the_word_flags_it():
    text = _section(3)
    assert _number(POOLS["detector_ungraded"]) in text and POOLS["detector_ungraded"] == 39936
    _expect(text, r"at least one detector[^\n]*\d", r"all four detectors[^\n]*\d",
            r"detector that graded")


def test_section_3_pins_the_draw_mechanics_without_drawing():
    text = _section(3)
    _expect(text, r"permutation", r"prefix", r"inclusion probabilit", r"seeded",
            r"seeded[^.\n]*re-ask|re-ask[^.\n]*seeded", r"20 items", r"first verdict",
            r"round 3[^.\n]*worklist")
    assert SUMMARY["catch_loss"] == {"marked": 150, "matched": 0,
                                     "worklist": "consonant2_readout/catch_loss/worklist.jsonl"}
    _expect(text, r"matched\s*0|0 of (the |round 2's )?150|none of (the |round 2's )?150")


# --- verdicts ---------------------------------------------------------------------------------------
def test_section_4_lists_the_five_verdicts_and_the_mistake_kinds():
    text = _section(4)
    _expect(text, r"correct", r"mistake", r"not sure", r"different text", r"unusable audio",
            r"letter", r"haraka", r"shaddah", r"skipped", r"added", r"س/ص/ث|س/ص/ث",
            r"excluded from (the )?rates", r"not_in_clip")


# --- attribution ------------------------------------------------------------------------------------
def test_section_5_defines_every_bucket_and_the_precedence():
    text = _section(5)
    for bucket in BUCKETS:
        assert re.search(rf"(?<![\w-]){re.escape(bucket)}(?![\w-])", text), bucket
    _expect(text, r"F1\s*(→|->)\s*F1-align\s*(→|->)\s*F-ref\s*(→|->)\s*F2",
            r"M1r\s*(→|->)\s*M1\s*(→|->)\s*M1u\s*(→|->)\s*M2\s*(→|->)\s*M3")


def test_section_5_pins_the_matching_rules_of_the_architect_decisions():
    text = _section(5)
    _expect(text, r"hop\s*\+\s*overlap", r"code-point", r"neighbou?r", r"len\(?[^)\n]*\)?\s*≥\s*4",
            r"sensitivity", r"setting check",
            r"withheld", r"ungraded[^.\n]*(F3|M3|not learnable)",
            r"sorted[^.\n]*group_index", r"group_key", r"≥\s*3",
            r"reference suspect", r"both",
            r"strict", r"balanced")


# --- the decision rule ------------------------------------------------------------------------------
def test_section_6_freezes_the_owner_s_numbers():
    text = _section(6)
    _expect(text, r"≥ 3 instances in ≥ 3 clips", r"≥ 10 words", r"follow-along first",
            r"no verified catch", r"≥ 50\s?%", r"model share", r"≥ 60 confirmed",
            r"direction", r"F2\s*\+\s*M2", r"F3\s*\+\s*M3", r"strict-mode allowance",
            r"half of F3", r"soft[^.\n]*target")


# --- estimates --------------------------------------------------------------------------------------
def test_section_7_names_the_estimates_and_keeps_the_two_sides_apart():
    text = _section(7)
    _expect(text, r"per 1,000 correctly recited", r"per 100 confirmed", r"ADR-0008", r"never pooled",
            r"bootstrap", r"95\s?%", r"S4[^.\n]*bound", r"not a headline",
            r"biased upward", r"not a decision input", r"without[^.\n]*first and last")


def test_section_7_counts_the_known_verified_mistakes_from_the_two_catch_loss_audits():
    counts = {}
    for round_ in (2, 3):
        path = READOUTS / f"consonant{round_}_readout" / "catch_loss" / "verdicts.jsonl"
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
                if line.strip()]
        counts[round_] = sum(1 for row in rows if row["heard"] == "mistake")
    assert counts == {2: 11, 3: 22}
    text = _section(7)
    _expect(text, r"\b33\b", r"\b11\b", r"\b22\b", r"look")


# --- the pilot resize rule --------------------------------------------------------------------------
def test_section_8_states_the_mechanical_resize_rule():
    text = _section(8)
    _expect(text, r"S1\s*≥\s*150", r"S4\s*≥\s*40", r"S5\s*≥\s*20", r"repeats\s*≥\s*15",
            r"highest[^.\n]*miss yield", r"ties[^.\n]*S3", r"\b60\b", r"3 h", r"~\s?42",
            r"direction", r"dated amendment", r"before the main sitting",
            r"2\.5 h", r"35[- ]min")


# --- dataset rows -----------------------------------------------------------------------------------
def test_section_9_fixes_the_dataset_row_format_and_its_limits():
    text = _section(9)
    _expect(text, r"tlog_mistakes\.jsonl", r"audio_path", r"shard", r"\brow\b", r"sample span",
            r"sha256", r"verdict", r"kinds", r"whole_played", r"CC-BY-4\.0", r"no audio",
            r"group_key", r"split", r"mined[^.\n]*tier", r"seed v0", r"#225", r"#257", r"Ship A")

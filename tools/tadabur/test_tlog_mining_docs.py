"""Acceptance tests for the tlog mining pre-registration documents (#262).

Committed before the documents exist and **locked**: later turns may add tests but never weaken
or rewrite these. The change is doc-only, so the tests guard the facts the documents pin, not
their prose: every identity, checksum, count and threshold a document quotes is recomputed from
the committed artifacts, and the frozen documents it amends must stay byte-identical.

Contract this file pins
-----------------------
* ``docs/tlog-mining-amendment.md`` (new, dated 2026-11-02): the mining rule for confirm.1/2.
  The frozen ``docs/tlog-holdout-preregistration.md`` is never edited.
* ``docs/tlog-mining-preregistration.md`` (new, dated 2026-11-02): the pilot's design, frozen
  before the draw.
* Checksums are quoted as full lowercase 64-hex SHA-256 strings; commits as full 40-hex SHAs
  (except 7d8320c, quoted as the verdict commit's prefix).
* The harness scoring key is quoted by name (``allowances_off``) and by its key fields
  (``softPairsEnabled`` and ``shaddahSuppression`` both false), recomputed here from
  ``training.muraja_policy``.
* The strata table has one row per stratum, the stratum label first and the word count last.
"""

import hashlib
import itertools
import json
import re
import subprocess
from pathlib import Path

import pytest

from training import muraja_policy

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
AMENDMENT = DOCS / "tlog-mining-amendment.md"
PREREG = DOCS / "tlog-mining-preregistration.md"
FROZEN = DOCS / "tlog-holdout-preregistration.md"
GATING = DOCS / "tlog-gating-amendment.md"
TRANCHES = ROOT / "tools" / "tadabur" / "tlog" / "tranches"
READOUT = ROOT / "tools" / "tadabur" / "consonant2_readout"
DECODES = READOUT / "decodes"

FROZEN_SHA256 = "204fe7f980488aff73402c56f1729f7f17f7c9471ea0664f6365bb8870b62086"
GATING_SHA256 = "23d6bf7abc9882909e9686d19b2bdcd8b19fb19ed49a38829c4db9e58a4ce085"
DATASET_REVISION = "9fe4f7e6d1b372c2c11157d6d2f0f91ca344eb51"
VERDICT_COMMIT_PREFIX = "7d8320c"
DATE = "2026-11-02"
SEED = "tlog-mining-pilot-2026-11"
DETECTOR_DECODES = ("base", "c95", "r2_candidate_ema")
#: stratum label (as the pre-registration's table words it) -> pilot size.
STRATA = {"all 3": 60, "exactly 2": 50, "only teacher": 25, "only C95": 25,
          "only round-2 EMA": 25, "none flag": 55}
PILOT_TOTAL = 240


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _text(path: Path) -> str:
    assert path.is_file(), f"{path.relative_to(ROOT)} is not committed"
    return path.read_text(encoding="utf-8")


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _gating_clips(role: str) -> set[str]:
    with (TRANCHES / f"{role}.jsonl").open(encoding="utf-8") as lines:
        rows = (json.loads(line) for line in lines)
        return {row["audio_path"] for row in rows if row["stratum"] == "gating"}


def _excluded_clip_keys() -> list[str]:
    return _json(READOUT / "report.json")["coverage"]["graded"]["excluded_clips"]["keys"]


def _identities() -> dict[str, dict]:
    return _json(DECODES / "meta.confirm.json")["identities"]


def _expect(text: str, *patterns: str) -> None:
    missing = [p for p in patterns if not re.search(p, text, re.IGNORECASE)]
    assert not missing, f"missing: {missing}"


def _strata_rows(text: str) -> dict[str, int]:
    """Each markdown table row whose last cell is a count, as ``{first cell: count}``."""
    rows = {}
    for line in text.splitlines():
        cells = [cell.strip().strip("*").strip() for cell in line.strip().strip("|").split("|")]
        if line.lstrip().startswith("|") and len(cells) >= 2 and cells[-1].isdigit():
            rows[cells[0]] = int(cells[-1])
    return rows


def _cap_draw(order: tuple[str, ...], clips: dict[str, str], *, quota: int, cap: int,
              taken: dict[str, int] | None = None) -> list[str]:
    """Walk a rank order under a per-clip cap, taking words until the quota.

    ``taken`` seeds the counters, so a test can model a clip shared across strata."""
    per_clip, drawn = dict(taken or {}), []
    for word in order:
        if len(drawn) == quota:
            break
        if per_clip.get(clips[word], 0) < cap:
            per_clip[clips[word]] = per_clip.get(clips[word], 0) + 1
            drawn.append(word)
    return drawn


# --- the frozen documents stay frozen -------------------------------------------------------------
def test_frozen_holdout_doc_is_unchanged():
    assert _sha256(FROZEN) == FROZEN_SHA256


def test_gating_amendment_is_unchanged():
    assert _sha256(GATING) == GATING_SHA256


# --- the amendment --------------------------------------------------------------------------------
def test_amendment_is_dated_and_frozen_and_amends_rather_than_edits():
    text = _text(AMENDMENT)
    _expect(text, DATE, r"frozen", r"amendment", r"#262", r"tlog-holdout-preregistration\.md")


def test_amendment_cites_the_frozen_documents_by_sha256():
    text = _text(AMENDMENT)
    assert FROZEN_SHA256 in text and GATING_SHA256 in text


def test_amendment_names_the_claims_and_the_verdict_it_requires():
    text = _text(AMENDMENT)
    for role in ("confirm.1", "confirm.2"):
        claim = TRANCHES / f"{role}.claim.json"
        assert f"{role}.claim.json" in text
        assert _sha256(claim) in text, f"{role}'s claim checksum"
        assert _json(claim)["commit"] in text, f"{role}'s pre-claim commit"
    assert "consonant2_readout/verdict.json" in text and VERDICT_COMMIT_PREFIX in text
    assert _sha256(READOUT / "verdict.json") in text


def _git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True, check=False)


def test_verdict_and_claims_were_committed_together_at_the_pinned_commit():
    added = _git("log", "--diff-filter=A", "--format=%H", "--",
                 str(TRANCHES / "confirm.1.claim.json"), str(TRANCHES / "confirm.2.claim.json"),
                 str(READOUT / "verdict.json"))
    if added.returncode:
        # Only a checkout without this history may skip: not a repository, or a shallow clone.
        shallow = _git("rev-parse", "--is-shallow-repository")
        if "not a git repository" in shallow.stderr.lower():
            pytest.skip("not a git checkout: the claim and verdict commits cannot be read")
        assert shallow.stdout.strip() == "true", added.stderr
        pytest.skip("shallow clone: the claim and verdict commits are not reachable")
    commits = added.stdout.split()
    assert commits, "no commit added the claims and the verdict"
    assert len(commits) == 1 and commits[0].startswith(VERDICT_COMMIT_PREFIX)


def test_amendment_states_the_mining_rule_and_its_consequences():
    _expect(_text(AMENDMENT),
            r"mined", r"claim", r"verdict",
            r"reported:", r"gating",
            r"never again a sealed", r"tlog\.training", r"#258",
            r"confirm\.3", r"confirm\.4", r"round 3")


def test_amendment_records_the_teacher_timing_pass_and_the_audio_rule():
    text = _text(AMENDMENT)
    _expect(text, r"teacher-only timing", r"not a student read", r"no claim", r"S7",
            r"public", r"no tlog audio", r"SHA-256")
    assert DATASET_REVISION in text


def test_amendment_records_the_owner_answers_verbatim():
    assert "#1 yes, #2 agreed, #3 yes, #4 yes" in _text(AMENDMENT)


# --- the pilot pre-registration -------------------------------------------------------------------
def test_prereg_is_dated_frozen_before_the_draw_and_cites_the_amendment():
    text = _text(PREREG)
    _expect(text, DATE, r"frozen", r"before the draw", r"tlog-mining-amendment\.md")


def test_population_counts_match_the_tranche_manifests():
    confirm1, confirm2 = _gating_clips("confirm.1"), _gating_clips("confirm.2")
    excluded = {key.split("#")[0] for key in _excluded_clip_keys()}
    kept = (confirm1 | confirm2) - excluded
    assert (len(confirm1), len(confirm2), len(excluded)) == (2_843, 2_919, 20)
    assert excluded <= confirm1 | confirm2 and not confirm1 & confirm2
    text = _text(PREREG)
    for count in (len(confirm1), len(confirm2), len(confirm1 | confirm2), len(kept)):
        assert f"{count:,}" in text, f"{count:,}"
    _expect(text, r"clean", r"gating", r"clip key", r"0-based", r"catch_loss_audit")
    # The draw keys a word by ``graded_words``' own pair, not by the canonical word index.
    assert "(`clip key`, `position`)" in text
    _expect(text, r"`position`", r"`word_index`", r"graded_words")


def test_exclusions_are_the_round_two_excluded_clips_pinned_by_sha256():
    text = _text(PREREG)
    assert _sha256(READOUT / "report.json") in text
    _expect(text, r"excluded_clips", r"#248", r"55:17", r"90:8", r"106:1")


def test_detectors_are_pinned_by_checksum_and_no_zipformer_is_used():
    identities = _identities()
    text = _text(PREREG)
    for model in DETECTOR_DECODES:
        decode = DECODES / f"tlog.confirm.{model}.jsonl.gz"
        assert _sha256(decode) in text, f"{decode.name} checksum"
    assert identities["tlog.confirm/base"]["identity"]["hub_revision"] in text
    for model in ("c95", "r2_candidate_ema"):
        assert identities[f"tlog.confirm/{model}"]["identity"]["checkpoint_sha256"] in text
    _expect(text, r"obadx/muaalem-model-v3_2", r"`?checks`? arm|arm.{0,20}`?checks`?",
            r"EMA", r"no zipformer")
    assert not any(DECODES.glob("tlog.confirm.*zipformer*")), "a zipformer decode appeared"


def test_round_three_candidate_is_not_a_detector_here():
    _expect(_text(PREREG), r"round-3 candidate", r"phase C", r"round 3's verdict")


def test_flag_is_the_harness_word_grade_with_every_allowance_off():
    allowances_off = next(s for s in muraja_policy.SCORINGS if s.name == "allowances_off")
    assert (allowances_off.mode, allowances_off.soft_pairs_enabled,
            allowances_off.shaddah_suppression) == ("balanced", False, False)
    text = _text(PREREG)
    assert muraja_policy.MURAJA_COMMIT in text
    assert allowances_off.tashkeel_shown is True
    sent = muraja_policy.engine_runs([allowances_off])[0]
    assert set(sent) == {"name", "mode", "soft_pairs_enabled", "shaddah_suppression"}, sent
    _expect(text, r"allowances_off", r"softPairsEnabled\W{0,8}false",
            r"shaddahSuppression\W{0,8}false", r"ADR-0012", r"`?flagged`?",
            r"shipped_default", r"never used for selection",
            r"displayed\(\)", r"not sent to the binary")


def test_strata_table_has_the_pilot_sizes_and_sums_to_240():
    rows = _strata_rows(_text(PREREG))
    sizes = {label: next((n for row, n in rows.items() if label.lower() in row.lower()), None)
             for label in STRATA}
    assert sizes == STRATA
    assert sum(STRATA.values()) == PILOT_TOTAL
    assert next((n for row, n in rows.items() if row.lower() == "total"), None) == PILOT_TOTAL


def test_draw_rules_are_stated_with_the_seed_and_inclusion_probability():
    _expect(_text(PREREG),
            re.escape(SEED), r"judgment call", r"salted hash",
            r"at most 2 words per clip", r"4 per `?group_key`?",
            r"shortfall", r"all 3", r"exactly 2", r"none", r"inclusion probabilit",
            r"actual sizes")


def test_cap_aware_weights_guard_the_unequal_clip_example():
    """Three words in one clip, quota 3, clip cap 2: marginals are 2/3 and 1, not 3/4."""
    clips = {"a": "A", "b": "A", "c": "A", "d": "B"}
    draws = [_cap_draw(order, clips, quota=3, cap=2) for order in itertools.permutations("abcd")]
    frequency = {word: sum(word in draw for draw in draws) / len(draws) for word in "abcd"}
    assert frequency == {"a": 2 / 3, "b": 2 / 3, "c": 2 / 3, "d": 1.0}
    sample = _cap_draw(("a", "b", "c", "d"), clips, quota=3, cap=2)
    assert sample == ["a", "b", "d"]
    assert sum(1 / frequency[word] for word in sample if word != "d") == pytest.approx(3)
    _expect(_text(PREREG), r"π_a = π_b = π_c = 2/3", r"not 3/4", r"R = 10,000", r"replay")


def test_each_stratum_draws_on_its_own_so_a_shared_clip_cannot_starve_one():
    """A shared cap counter would skip the second stratum's word forever; per-stratum ones do not."""
    filled = _cap_draw(("a", "b"), {"a": "A", "b": "A"}, quota=2, cap=2)
    assert filled == ["a", "b"]
    # The first stratum's two words exhaust clip A for every later stratum under one counter.
    assert _cap_draw(("x",), {"x": "A"}, quota=3, cap=2, taken={"A": 2}) == []
    # Drawn on its own, the second stratum's word is always taken: π_x = 1.
    assert _cap_draw(("x",), {"x": "A"}, quota=1, cap=2) == ["x"]
    _expect(_text(PREREG), r"each stratum is drawn on its own", r"independent draws",
            r"positive inclusion probability", r"`c_w = 0`", r"unresolved")


def test_presentation_is_the_reference_aligned_snip_with_a_whole_clip_fallback():
    _expect(_text(PREREG),
            r"excerpt_span", r"0\.3 s", r"one word either side", r"forced alignment",
            r"reference", r"never from a decoded mark", r"play whole clip", r"fallback rate")


def test_question_kinds_and_verified_mistake_definition():
    _expect(_text(PREREG),
            r"Was this word recited correctly\?", r"yes", r"unclear", r"ADR-0007",
            r"wrong letter", r"haraka", r"shaddah", r"missing or extra", r"tajweed only",
            r"other", r"verified mistake", r"decisive")


def test_readout_is_reported_only_with_the_prescribed_statistics():
    _expect(_text(PREREG),
            r"reported only", r"Wilson", r"Horvitz", r"bootstrap", r"group_key",
            r"س", r"ص", r"ث", r"seconds per word", r"ADR-0008", r"never pooled")


def test_scale_up_rule_and_later_training_use():
    _expect(_text(PREREG),
            r"scale-up", r"15\s?%", r"\b20 verified mistakes", r"re-plan",
            r"split by `?group_key`?", r"eval and training halves")

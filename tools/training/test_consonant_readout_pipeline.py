"""The consonant round's pipeline: #95 regenerated, the reproduction check, freeze and G6 (#175).

The doc is byte-identical, the report is key-for-key except RetaSy's restricted subtree (absent
here, #171), and the reproduction check reveals aggregates only. G6 reads the truth baseline's
caches read-only; the tracer reports it as "cannot certify" until C95's caches exist (#176).
"""

from __future__ import annotations

import json

import pytest

from tadabur.qud_text import ItemReference
from training import aug_readout as ar
from training import aug_readout_doc as adoc
from training import aug_readout_gate as ag
from training import aug_readout_spec as asp
from training import consonant_readout as cr
from training import truth_baseline as tb
from training.external_corpus import WordGrade
from training.muraja_policy import SHIPPED_DEFAULT, SiteGrade, WORD_FLAGGED
from training.nonpro_eval import ClipItem
from training.word_flags import Flag

SHIPPED = SHIPPED_DEFAULT.name
C95_CHECKPOINT = "8490b63d0664827d5a2e2e3121eb4af7b22069273db39268c7638cc7d8839cd6"


def _harness() -> bool:
    from training.muraja_policy import Harness

    return Harness.locate() is not None


def _skip_without_harness() -> None:
    if not _harness():
        pytest.skip("the Muraja harness is not built here")


# --- #95's committed readout from the aug95 spec -----------------------------------------------


def test_the_aug95_doc_regenerates_byte_identically():
    root = ar.READOUT_DIR
    look = json.loads((root / "look.json").read_text(encoding="utf-8"))
    report = json.loads((root / "report.json").read_text(encoding="utf-8"))
    coverage = json.loads((root / "coverage.json").read_text(encoding="utf-8"))
    assert adoc.render(look, report, coverage, spec=asp.AUG95) == ar.DOC_PATH.read_text(
        encoding="utf-8")


def _tables(report: dict) -> dict:
    """The report's tables: the sitting's audit fields and RetaSy's restricted subtree aside."""
    tables = json.loads(json.dumps(report))
    for key in ("audit", "baselines", "choice", "issue"):
        tables.pop(key, None)
    tables["catch"].pop("retasy_in_correct", None)
    return tables


def test_the_aug95_report_regenerates_key_for_key_except_the_retasy_subtree():
    _skip_without_harness()
    if not (ar.READOUT_DIR / "restricted").exists():
        pytest.skip("#171: RetaSy's restricted decodes are not present")
    regenerated = ar.confirm_report(ar.load_confirm(ar.Sources(), "ema", spec=asp.AUG95),
                                    spec=asp.AUG95)
    committed = json.loads((ar.READOUT_DIR / "report.json").read_text(encoding="utf-8"))
    assert "retasy_in_correct" in committed["catch"]  # taken from the committed file
    assert ag.canonical(_tables(regenerated)) == ag.canonical(_tables(committed))


# --- the reproduction check --------------------------------------------------------------------


def test_the_reproduction_check_reproduces_95_s_published_totals_and_writes_nothing(monkeypatch):
    _skip_without_harness()

    def refuse(*args, **kwargs):
        raise AssertionError("the reproduction check wrote a file")

    monkeypatch.setattr(ar, "write_text_atomically", refuse)
    counts = ar.reproduction(ar.Sources())
    assert ar.REPRODUCTION == {"tarteel_v1_confirm": {"flagged": 1971, "graded": 41572},
                               "qdat": {"flagged": 670, "graded": 11502}}
    assert ar.REPRODUCTION_MODELS == ("candidate_ema", "control_ema", "h448", "base")
    assert counts == ar.REPRODUCTION
    for row in counts.values():  # totals only: never a word, a cause or a substitution
        assert set(row) == {"flagged", "graded"}


def test_a_perturbed_reproduction_count_aborts():
    ar.require_reproduction(ar.REPRODUCTION)
    for path, key in (("tarteel_v1_confirm", "flagged"), ("qdat", "graded")):
        perturbed = json.loads(json.dumps(ar.REPRODUCTION))
        perturbed[path][key] += 1
        with pytest.raises(SystemExit, match="reproduc"):
            ar.require_reproduction(perturbed)


# --- freeze: the look half only -----------------------------------------------------------------


def _item(n: int) -> ClipItem:
    name = f"c{n:03d}.wav"
    return ClipItem(f"{name}#0", "qdat", name, 5, 109, 0, 16000, "e" * 64,
                    ItemReference(("قَالَ", "سَططَ"), (7, 8), "قَالَ سَططَ", (0, 6)),
                    f"r{n}", n, {"final_label": "in_correct"})


def _grade(model: str, item: ClipItem, position: int, flag: Flag | None) -> WordGrade:
    return WordGrade(model, SHIPPED, item, position, 2, position + 1, 7 + position,
                     SiteGrade("flagged" if flag is not None else "not_flagged"), flag)


def _freeze_grades(items: list[ClipItem]) -> list[WordGrade]:
    consonant = Flag(("other_consonant",), (), "minor", (),
                     consonants=(("other_consonant", "س→ش"),))
    tashkeel = Flag(("tashkeel",), ("fatha dropped",), "tashkeel", ())
    flags = [consonant, tashkeel, None, None, None, None]
    return [_grade("c95", item, position, flags[2 * i + position])
            for i, item in enumerate(items) for position in (0, 1)]


class LookHalf:
    """The look half's synthetic decodes and grades; it refuses to be read as anything else."""

    def __init__(self, items: list, grades: list):
        self.items, self.grades, self.read = items, grades, []

    def tarteel(self, which_half: str, models) -> ar.SetData:
        assert which_half == "look"
        self.read.append(which_half)
        return ar.SetData(self.items,
                          {m: {i.key: {} for i in self.items} for m in models})

    def grade(self, items, decodes, models) -> list:
        return self.grades


class ConfirmHalf:
    """A sources that fails if the confirm half or its grades are read."""

    def tarteel(self, *args, **kwargs):
        raise AssertionError("the confirm half was read")

    def grade(self, *args, **kwargs):
        raise AssertionError("the confirm half was graded")


def test_freeze_refuses_the_confirm_half_always():
    with pytest.raises(SystemExit, match="confirm"):
        cr.run_freeze(asp.CONSONANT1, ConfirmHalf(), half="confirm")


def test_freeze_dry_runs_on_the_look_half_without_writing_freeze_json():
    items = [_item(i) for i in range(3)]
    before = (cr.ROUND_DIR / "freeze.json").exists()
    sources = LookHalf(items, _freeze_grades(items))
    record = cr.run_freeze(asp.CONSONANT1, sources, half="look")
    assert sources.read == ["look"]
    assert record["model"] == "c95"  # the round's baseline
    assert record["causes"] == {"consonant": 1, "tashkeel": 1}
    assert record["substitutions"] == [{"cause": "other_consonant",
                                        "substitution": "س→ش", "words": 1}]
    assert (cr.ROUND_DIR / "freeze.json").exists() == before  # #177's artifact, not #175's


# --- the round's models and the frozen G6 sites -------------------------------------------------


def test_the_round_s_models_json_holds_c95_s_identity():
    records = json.loads(cr.MODELS_PATH.read_text(encoding="utf-8"))
    identity = ag.validate_identity("qdat", "c95", records["qdat/c95"], spec=asp.CONSONANT1)
    assert identity["checkpoint_sha256"] == C95_CHECKPOINT


def test_the_frozen_g6_sites_are_the_eight_held_out_mistakes_at_the_two_pairs():
    sites = cr.g6_sites()
    assert len(sites) == 8
    assert {site.mark for site in sites} == {"س↔ش", "ط↔ق"}
    assert all(site.heard != site.prescribed for site in sites)
    assert cr.load_g6_sites() == [site.site_id for site in sites]


def test_a_model_without_a_check_cache_leaves_the_catch_arm_unread(tmp_path):
    assert cr.catch_arm(cr.g6_sites(), ("h448",), cache_dir=tmp_path) is None


def _truth_baseline_outcomes(sites) -> dict[str, dict[str, str]]:
    from tadabur.staged_audio import REGISTRY_PATH, load_staged_clips

    registry = load_staged_clips(REGISTRY_PATH)
    items = tb.items_of(sites, registry)
    out = {}
    for name in ("h448", "base"):
        _, recorded = tb.model_caches(name, tb.DEFAULT_MODELS[name], items, registry, None)
        sequences = {name: {k: v[tb.CHECKS] for k, v in recorded["items"].items()
                            if k in items}}
        decodes = {name: {k: tb.committed_text(c) for k, c in sequences[name].items()}}
        _, kept = tb.app_outcomes(sites, [], decodes, sequences, tb.require_harness())
        out[name] = {site.site_id: kept.grades[SHIPPED][name][site.site_id].outcome
                     for site in sites}
    return out


def test_g6_plumbing_reproduces_truth_baseline_s_outcomes_on_the_eight_sites():
    _skip_without_harness()
    sites = cr.g6_sites()
    arm = cr.catch_arm(sites, ("h448", "base"))
    assert arm is not None
    assert arm["sites"] == [site.site_id for site in sites]
    expected = _truth_baseline_outcomes(sites)
    for name in ("h448", "base"):
        assert arm["models"][name]["outcomes"] == expected[name]
        assert arm["models"][name]["flagged"] == sum(
            outcome in WORD_FLAGGED for outcome in expected[name].values())
        assert arm["models"][name]["graded"] == len(sites)
    assert arm["difference"] is None  # candidate and baseline are not both read

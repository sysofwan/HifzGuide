"""The consonant round's pipeline: #95 regenerated, the reproduction check, freeze and G6 (#175).

The doc is byte-identical, the report is key-for-key except RetaSy's restricted subtree (absent
here, #171), and the reproduction check reveals aggregates only. G6 reads the truth baseline's
caches read-only; the tracer reports it as "cannot certify" until C95's caches exist (#176).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from tadabur.qud_text import ItemReference
from training import aug_readout as ar
from training import aug_readout_doc as adoc
from training import aug_readout_gate as ag
from training import aug_readout_metrics as am
from training import aug_readout_spec as asp
from training import aug_readout_verdict as av
from training import consonant_readout as cr
from training import truth_baseline as tb
from training.external_corpus import WordGrade, flag_view
from training.muraja_policy import SHIPPED_DEFAULT, SiteGrade, WORD_FLAGGED
from training.nonpro_eval import ClipItem
from training.word_flags import CAUSES, Flag

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


def test_freeze_dry_runs_on_the_look_half_and_writes_neither_artifact(monkeypatch, tmp_path,
                                                                      capsys):
    """The CLI's dry run prints the summary and leaves both artifacts alone (review R1.2):
    absent when they never existed, byte-identical when they do."""
    items = [_item(i) for i in range(3)]
    sources = LookHalf(items, _freeze_grades(items))
    record_path, doc_path = tmp_path / "freeze.json", tmp_path / "freeze.md"
    monkeypatch.setattr(cr, "FREEZE_PATH", record_path)
    monkeypatch.setattr(cr, "FREEZE_DOC_PATH", doc_path)
    monkeypatch.setattr(ar, "Sources", lambda spec: sources)
    monkeypatch.setattr(sys, "argv", ["consonant_readout", "freeze", "--half", "look",
                                      "--dry-run"])

    cr.main()
    record = json.loads(capsys.readouterr().out)
    assert sources.read == ["look"]
    assert record["model"] == "c95"  # the round's baseline
    assert record["causes"] == {"consonant": 1, "tashkeel": 1}
    assert record["substitutions"] == [{"cause": "other_consonant",
                                        "substitution": "س→ش", "words": 1}]
    assert not record_path.exists() and not doc_path.exists()

    record_path.write_text('{"kept": 1}\n', encoding="utf-8")
    doc_path.write_text("kept\n", encoding="utf-8")
    cr.main()
    assert record_path.read_text(encoding="utf-8") == '{"kept": 1}\n'
    assert doc_path.read_text(encoding="utf-8") == "kept\n"


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


# --- #95's report from the committed data alone (self-check r2, S2.7) --------------------------


def _synthetic_retasy() -> ar.SetData:
    """One RetaSy-shaped item graded for every role: its tables come from the committed report,
    but the other tables must not need restricted decodes to be regenerated (#171)."""
    item = _item(0)
    grades = [_grade(role, item, 0, None) for role in ar.ROLES]
    return ar.SetData([item], {role: {} for role in ar.ROLES}, grades)


class _NoRetasy:
    """Sources whose RetaSy set is synthetic (its decodes are absent here, #171); every other
    set is the real one, read from the committed decodes."""

    def __init__(self, inner):
        self._inner = inner
        self._retasy = _synthetic_retasy()

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def retasy(self, models) -> ar.SetData:
        return ar.SetData(list(self._retasy.items), {model: {} for model in models})

    def grade(self, items, decodes, models) -> list:
        if items and items[0] is self._retasy.items[0]:
            return self._retasy.grades
        return self._inner.grade(items, decodes, models)


def test_the_aug95_report_regenerates_key_for_key_without_the_restricted_decodes():
    _skip_without_harness()
    data = ar.load_confirm(_NoRetasy(ar.Sources()), "ema", spec=asp.AUG95)
    regenerated = ar.confirm_report(data, spec=asp.AUG95)
    committed = json.loads((ar.READOUT_DIR / "report.json").read_text(encoding="utf-8"))
    assert "retasy_in_correct" in committed["catch"]  # taken from the committed file
    for key in ("audit", "baselines", "choice", "issue"):
        regenerated.pop(key, None)
        committed.pop(key, None)
    # RetaSy's restricted decodes are #171's; its subtree is the committed file's.
    regenerated["catch"]["retasy_in_correct"] = committed["catch"]["retasy_in_correct"]
    regenerated["sizes"]["retasy_in_correct"] = committed["sizes"]["retasy_in_correct"]
    assert ag.canonical(regenerated) == ag.canonical(committed)


# --- the round's spec-driven requirements (self-check r2, S2.3) --------------------------------


def test_the_consonant_round_never_demands_the_control_s_decodes():
    consonant = ag.needs("confirm", "ema", asp.CONSONANT1)
    assert all("control_ema" not in models for models in consonant.values())
    assert consonant["qdat"] == ("candidate_ema", "c95")
    assert consonant["tadabur_dev_unseen"] == ("candidate_ema", "c95", "h448", "base")
    assert "control_ema" in ag.needs("confirm", "ema", asp.AUG95)["qdat"]  # #95 unchanged
    files = ar.Sources(asp.CONSONANT1).input_files("confirm", "ema")
    assert not any("control" in path.name for path in files)


def test_the_reported_qud_half_is_a_declared_input_of_the_sitting():
    """Reporting-only means "not in the verdict", never "outside provenance" (review R1.6)."""
    assert ag.needs("confirm", "ema", asp.CONSONANT1)["qud_half"] == ("candidate_ema", "c95")
    files = ar.Sources(asp.CONSONANT1).input_files("confirm", "ema")
    assert asp.CONSONANT_ROOT / "decodes" / "qud_half.candidate_ema.jsonl.gz" in files
    assert asp.AUG95_ROOT / "decodes" / "qud_half.candidate_ema.jsonl.gz" in files
    # The merge's count table covers every declared set, so a supplied identity cannot KeyError.
    counts = ar.expected_counts(asp.CONSONANT1)
    assert {name for name in asp.CONSONANT1.sets if name in ag.SETS} <= set(counts)
    assert counts["qud_half"] == len(ar.load_qud_selection())
    # The merge writes the candidate's half under the round's root; C95's stays #95's file, so
    # a supplied candidate half is neither overwritten nor ever read as C95's (R1.6/R1.1).
    sources = ar.Sources(asp.CONSONANT1)
    assert sources.decode_path("qud_half", "candidate_ema") == (
        asp.CONSONANT_ROOT / "decodes" / "qud_half.candidate_ema.jsonl.gz")
    assert sources.decode_path("qud_half", asp.C95) == (
        asp.AUG95_ROOT / "decodes" / "qud_half.candidate_ema.jsonl.gz")


def test_the_sitting_lists_the_freeze_and_the_caches_it_reads(monkeypatch, tmp_path):
    freeze = tmp_path / "freeze.json"
    freeze.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cr, "FREEZE_PATH", freeze)
    files = ar.Sources(asp.CONSONANT1).input_files("confirm", "ema")
    assert freeze in files
    from training.truth_baseline import CHECK_STORE, DECODE_STORE

    for store in (CHECK_STORE, DECODE_STORE):
        if store.path("c95").exists():
            assert store.path("c95") in files


# --- the freeze's list is the round's consonant causes (self-check r2, S2.4) -------------------


def test_the_freeze_substitution_list_holds_only_the_round_s_consonant_causes():
    item = _item(0)
    flags = [Flag(("other_consonant",), (), "minor", (),
                  consonants=(("other_consonant", "ا→∅"),)),
             Flag(("shaddah",), (), "minor", (), consonants=(("shaddah", "ل→∅"),)),
             Flag(("target_pair",), (), "minor", (), consonants=(("target_pair", "ط→ق"),)),
             Flag(("soft_pair",), (), "minor", (), consonants=(("soft_pair", "ن→ل"),))]
    record = cr.freeze_report([_grade("c95", item, i, flag) for i, flag in enumerate(flags)],
                              "c95")
    assert {row["substitution"] for row in record["substitutions"]} == {"ط→ق", "ن→ل"}
    assert record["causes"]["consonant"] == 2


# --- the catch arm's pairing, frozen ids and scope (self-check r2, S2.1/S2.2/S2.12) ------------


class _Site:
    def __init__(self, site_id: str):
        self.site_id = site_id


def test_the_catch_table_pairs_the_candidate_against_the_baseline_it_reads():
    sites = [_Site(f"a{i}") for i in range(3)]
    outcomes = {"candidate_ema": {"a0": "flagged", "a1": "not_flagged", "a2": "flagged"},
                "c95": {"a0": "flagged", "a1": "flagged", "a2": "not_flagged"}}
    table = cr.catch_table(sites, outcomes, asp.C95)
    assert [table["models"][name]["flagged"] for name in outcomes] == [2, 2]
    assert table["difference"] == 0  # 2 − 2: the earlier wiring read None here
    assert cr.catch_table(sites, {"h448": outcomes["c95"]}, asp.C95)["difference"] is None


def _check_cache(path, name: str, ref: str, identity: dict) -> None:
    """A truth-baseline check cache of ``ref`` with the given identity and no items."""
    path.write_text(json.dumps({
        "model": name, "identity": identity,
        "fingerprints": {protocol: tb.expected_fingerprint(ref, mode)
                         for protocol, mode in tb.CHECK_PROTOCOLS.items()},
        "items": {}}), encoding="utf-8")


def test_a_cache_whose_identity_is_not_the_round_s_pin_is_refused():
    """G6 must not decide against other weights (review R1.2)."""
    expected = {"model_ref": asp.C95_REF, "weights": "ema", "step": 10_000,
                "recipe": "aug94-mixture-v2", "checkpoint_sha256": asp.C95_SHA256}
    # One checkpoint holds both selectors: a selector-free identity is no evidence (R2.1).
    assert not cr.binds_cache({"model_ref": asp.C95_REF, "checkpoint_sha256": asp.C95_SHA256},
                              expected)
    for edit in ({"weights": "live"}, {"weights": None}, {"step": 1}, {"recipe": "wrong"},
                 {"checkpoint_sha256": "0" * 64}):
        assert not cr.binds_cache({**expected, **edit}, expected)
    assert cr.binds_cache({**expected}, expected)
    # A live pin accepts the legacy selector-free identity: absence means live (R3.1).
    live = {**expected, "weights": tb.LIVE}
    assert cr.binds_cache({"model_ref": asp.C95_REF,
                           "checkpoint_sha256": asp.C95_SHA256}, live)
    assert not cr.binds_cache({"model_ref": asp.C95_REF, "checkpoint_sha256": asp.C95_SHA256,
                               "weights": tb.EMA}, live)


def test_the_cache_producer_records_which_weights_it_decoded(tmp_path):
    """A SHA cannot select live or EMA, so the producer records EMA; live stays the legacy,
    selector-free shape every historical cache and identity already has (R2.1, R3.1)."""
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"weights")
    legacy = {"model_ref": str(checkpoint), "checkpoint_sha256": tb.file_sha256(checkpoint)}
    assert tb.model_identity(str(checkpoint), object()) == legacy
    assert tb.model_identity(str(checkpoint), object(), use_ema=True) == {
        **legacy, "weights": tb.EMA}
    assert tb.model_identity(str(checkpoint)) == legacy


class _FakeDecoder:
    """A decoder that supplies only what the cache code compares: the frozen fingerprints."""

    def __init__(self, ref: str):
        self.ref = ref

    def fingerprint(self, mode: str):
        from types import SimpleNamespace

        return SimpleNamespace(as_dict=lambda: tb.expected_fingerprint(self.ref, mode))


def test_extending_a_legacy_baseline_cache_keeps_its_identity(tmp_path, monkeypatch):
    """R3.1: a committed h448 cache (no selector) still accepts one new item."""
    from types import SimpleNamespace

    checkpoint = tmp_path / "h448.pt"
    checkpoint.write_bytes(b"weights")
    ref = str(checkpoint)
    store = tb.Store(tmp_path / "cache", tb.CHECK_PROTOCOLS, lambda decoder, span: {})
    legacy = tb.model_identity(ref)
    path = store.path("h448")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(tb.cache_text({
        "model": "h448", "identity": legacy,
        "fingerprints": {p: tb.expected_fingerprint(ref, m)
                         for p, m in store.protocols.items()},
        "items": {}}, store.one_item_per_line), encoding="utf-8")
    monkeypatch.setattr(tb.Decoder, "load", classmethod(
        lambda cls, model_ref, device, **kw: _FakeDecoder(model_ref)))
    monkeypatch.setattr(tb, "decode_items", lambda decoder, missing, registry, audio_dir,
                                              decode: {
        key: {"audio_sha256": site.audio_sha256} for key, site in missing.items()})
    cache = tb.update_cache(store, "h448", ref,
                            {"k0": SimpleNamespace(audio_sha256="a" * 64)}, {}, tmp_path)
    assert cache["identity"] == legacy and "k0" in cache["items"]


def test_resuming_an_external_decode_accepts_its_legacy_identity(tmp_path, monkeypatch):
    """R3.1: the h448 identity record a previous run wrote still matches this one."""
    from training import decoding, external_decode as ext

    checkpoint = tmp_path / "h448.pt"
    checkpoint.write_bytes(b"weights")
    record = {"model": "h448", "identity": tb.model_identity(str(checkpoint)),
              "fingerprints": {arm: tb.expected_fingerprint(str(checkpoint), mode)
                               for arm, mode in ext.MODES.items()}}
    (tmp_path / "h448.identity.json").write_text(
        json.dumps(record, indent=1) + "\n", encoding="utf-8")
    monkeypatch.setattr(decoding.Decoder, "load", classmethod(
        lambda cls, model_ref, device, **kw: _FakeDecoder(model_ref)))
    decoders = ext.load_decoders(lambda name: tmp_path / f"{name}.identity.json",
                                 models={"h448": ext.ModelSpec(str(checkpoint))})
    assert decoders["h448"].ref == str(checkpoint)


def test_a_fresh_default_baseline_identity_validates_against_the_committed_one(monkeypatch):
    """R3.1: the default (live) producer emits exactly the committed baseline identity."""
    from types import SimpleNamespace

    committed = ag.baseline_identities()["h448"]
    monkeypatch.setattr(tb, "Path", lambda ref: SimpleNamespace(is_file=lambda: True))
    monkeypatch.setattr(tb, "file_sha256", lambda path: committed["checkpoint_sha256"])
    produced = tb.model_identity(tb.DEFAULT_MODELS["h448"], object())
    assert produced == committed
    record = {"model": "h448",
              "fingerprints": ag.expected_fingerprints(tb.DEFAULT_MODELS["h448"]),
              "identity": produced}
    assert ag.validate_identity("tadabur_dev_unseen", "h448", record) == committed


def test_the_catch_arm_refuses_a_c95_cache_that_is_not_the_pinned_checkpoint(tmp_path):
    sites = cr.g6_sites()[:1]
    tampered = {"model_ref": asp.C95_REF, "weights": "live", "step": 1, "recipe": "wrong",
                "checkpoint_sha256": "0" * 64}
    _check_cache(tmp_path / "c95.json", "c95", asp.C95_REF, tampered)
    with pytest.raises(ValueError, match="not the round"):
        cr.catch_arm(sites, ("c95",), cache_dir=tmp_path)
    _check_cache(tmp_path / "c95.json", "c95", asp.C95_REF,
                 {"model_ref": asp.C95_REF, "weights": "ema", "step": 10_000,
                  "recipe": "aug94-mixture-v2", "checkpoint_sha256": asp.C95_SHA256})
    with pytest.raises(SystemExit, match="lacks"):  # the identity bound; the items are missing
        cr.catch_arm(sites, ("c95",), cache_dir=tmp_path)


def test_the_catch_arm_binds_the_candidate_s_cache_to_the_sitting_s_identity(tmp_path):
    from training.aug_readout_gate import ARMS

    ref = ARMS["candidate"]
    identity = {"model_ref": ref, "weights": "ema", "step": 10_000,
                "recipe": "aug94-mixture-v2", "checkpoint_sha256": "a" * 64}
    _check_cache(tmp_path / "candidate_ema.json", "candidate_ema", ref, identity)
    sites = cr.g6_sites()[:1]
    with pytest.raises(ValueError, match="no identity"):
        cr.catch_arm(sites, ("candidate_ema",), cache_dir=tmp_path)
    with pytest.raises(SystemExit, match="lacks"):
        cr.catch_arm(sites, ("candidate_ema",), cache_dir=tmp_path,
                     identities={"candidate_ema": identity})


def test_the_catch_arm_declares_both_cache_files(tmp_path):
    """R1.6: the required files are named even when absent."""
    assert cr.catch_required(("candidate_ema",), cache_dir=tmp_path) == (
        tmp_path / "candidate_ema.json",)
    assert set(cr.catch_required(("candidate_ema",))) == {
        tb.CHECK_STORE.path("candidate_ema"), tb.DECODE_STORE.path("candidate_ema")}


def test_a_missing_catch_cache_is_named_in_the_report_and_doc(tmp_path):
    missing = str(tmp_path / "candidate_ema.json")
    data = cr.RoundData(tarteel=None, qdat=None, qud_other=None, qud_half=None,
                        tadabur=None, bench={}, catch=None, catch_missing=(missing,))
    report = cr.confirm_report(data, spec=asp.CONSONANT1)
    assert report["missing_inputs"] == [missing]
    text = adoc.render(_look(), report, _round_coverage(), spec=asp.CONSONANT1)
    assert missing in text and "G6 cannot certify" in text


def test_the_confirm_sitting_reads_the_frozen_g6_sites_and_refuses_drift(monkeypatch):
    frozen = cr.load_g6_sites()
    assert [site.site_id for site in cr.g6_scope()["sites"]] == frozen
    live = cr.g6_sites()
    monkeypatch.setattr(cr, "_held_out_mistakes", lambda: live[:-1])
    with pytest.raises(SystemExit, match="denominator moved"):
        cr.g6_scope()


def test_the_frozen_g6_record_discloses_what_the_exclusions_removed():
    record = json.loads(cr.G6_SITES_PATH.read_text(encoding="utf-8"))
    scope = cr.g6_scope()
    assert record["candidates"] == scope["candidates"]
    assert record["excluded"] == scope["excluded"]
    assert record["sites"] == [site.site_id for site in scope["sites"]]


# --- the tracer runs #95's reproduction check (self-check r2, S2.8) ---------------------------


def test_the_tracer_runs_the_reproduction_check_and_prints_totals(monkeypatch):
    monkeypatch.setattr(ar, "reproduction", lambda sources: ar.REPRODUCTION)
    monkeypatch.setattr(cr, "tracer_views", lambda sources, workers: {})
    record = cr.tracer(object(), workers=1)
    assert record["reproduction"] == ar.REPRODUCTION
    text = cr.tracer_text(record["views"], reproduction_totals=record["reproduction"])
    assert "1971/41572" in text and "670/11502" in text
    assert "the #95 reproduction check" in text


def test_the_tracer_aborts_on_a_perturbed_reproduction(monkeypatch):
    perturbed = json.loads(json.dumps(ar.REPRODUCTION))
    perturbed["qdat"]["flagged"] += 1
    monkeypatch.setattr(ar, "reproduction", lambda sources: perturbed)
    monkeypatch.setattr(cr, "tracer_views", lambda sources, workers: {})
    with pytest.raises(SystemExit, match="reproduc"):
        cr.tracer(object())


# --- the consonant document (self-check r2, S2.5/S2.6) -----------------------------------------


def _render_interval() -> dict:
    return {"lower": None, "upper": None, "method": "none", "reason": "sparse or degenerate"}


def _view(points: dict, scale: float = am.POINTS, paired: bool = True,
          veto: bool = False) -> dict:
    comparisons = {}
    if paired:
        row = {"point": 0.0, "interval": _render_interval()}
        if veto:
            row["veto_interval"] = _render_interval()
        comparisons["candidate−c95"] = row
    return {"scale": scale, "observations": 4, "clusters": 2,
            "models": {name: {"point": point, "num": 1, "den": 2,
                              "interval": _render_interval()}
                       for name, point in points.items()},
            "comparisons": comparisons}


def _look() -> dict:
    return {"items": 4, "rule": "fewer false flags", "choice": "ema",
            "inputs": {"candidate_live": {"graded_words": 4, "flagged_words": 1,
                                          "false_flags_per_1000": 250.0,
                                          "word_agreement_points": 90.0}}}


def _catch() -> dict:
    return {"sites": ["s0", "s1", "s2"],
            "models": {"candidate_ema": {"graded": 3, "flagged": 1, "outcomes": {}},
                       "c95": {"graded": 3, "flagged": 2, "outcomes": {}}},
            "difference": -1,
            "scope": {"candidates": 3, "excluded": []}}


def _round_report(catch: dict) -> dict:
    students = {"candidate": 90.0, "c95": 89.0, "h448": 88.0, "base": 99.0}
    report = {
        "issue": asp.CONSONANT1.issue,
        "main": {"all_cause": _view({"candidate": 40.0, "c95": 47.41, "h448": 57.0}),
                 "consonant_cause": _view({"candidate": 10.0, "c95": 13.0, "h448": 15.0}),
                 "tashkeel_cause": _view({"candidate": 5.2, "c95": 5.0, "h448": 6.0}),
                 "qud_other_tashkeel_cause": _view({"candidate": 3.0, "c95": 3.1,
                                                    "h448": 4.0})},
        "mechanism": {name: _view(students) for name in (
            "tarteel_v1_confirm", "qdat", "tarteel_v1_confirm_departs", "qdat_departs",
            "tarteel_v1_confirm_equals", "qdat_equals")},
        "consistency": {"qdat": _view({"candidate": 60.0, "c95": 64.0, "h448": 70.0},
                                      scale=am.PER_1000)},
        "guards": {"qud_other": _view({"candidate": 4.0, "c95": 3.9, "h448": 5.0}),
                   "qud_half": _view({"candidate": 4.5, "c95": 4.6, "h448": 5.5}),
                   "tadabur_dev_unseen": _view({"candidate": 94.0, "c95": 94.1, "h448": 93.0,
                                                "base": 95.0}),
                   "qdat_bench": {kind: _view({"candidate": 90.0, "c95": 90.0, "h448": 89.0})
                                  for kind in ("consonant", "haraka", "shaddah")}},
        "substitutions": {"س→ش": _view({"candidate": 1.0, "c95": 1.2, "h448": 2.0},
                                       scale=am.PER_1000, veto=True)},
        "catch": {"ship_a_verified": catch,
                  "qdat_bench_departures": _view({"candidate": 5.0, "c95": 4.0, "h448": 6.0},
                                                 scale=am.PER_1000)},
        "leniency": {"blank_ratio": _view({"candidate": 1.0, "c95": 1.0, "h448": 2.0}),
                     "abstentions": {name: _view(students) for name in (
                         "tarteel_v1_confirm", "qdat")},
                     "ungraded": {name: _view({"candidate": 1.0, "c95": 2.0, "h448": 3.0},
                                              scale=am.PER_1000) for name in (
                         "tarteel_v1_confirm", "qdat", "qud_other")}},
        "sizes": {"tarteel_v1_confirm": 4, "qdat": 3, "qud_other": 2, "qud_half": 5,
                  "tadabur_dev_unseen": 6, "qdat_bench": 3, "ship_a_verified": len(catch["sites"])},
    }
    report["verdict"] = av.verdict(report, spec=asp.CONSONANT1)
    return report


def _population(evaluated: int) -> dict:
    return {"decoded": evaluated + 1, "evaluated": evaluated, "excluded": 1,
            "excluded_words": 2, "excluded_keys": ["x"], "reason": "a reason"}


def _round_coverage() -> dict:
    return {"issue": asp.CONSONANT1.issue, "statement": "The reading covers these populations.",
            "populations": {"tarteel_v1.look": _population(4),
                            "tarteel_v1.confirm": _population(4),
                            "qud_other": _population(2)}}


# --- reported substitutions and mistake marks are not the gating lists (review R1.4) -----------


def test_an_observed_substitution_outside_the_frozen_list_is_reported_not_gated():
    report = _round_report(_catch())
    report["substitutions"]["ق→ك"] = report["substitutions"]["س→ش"]
    report["g5_frozen"] = ["س→ش"]
    found = av.verdict(report, spec=asp.CONSONANT1)
    names = {condition["name"] for condition in found["conditions"]}
    assert "g5.س→ش" in names and "g5.ق→ك" not in names
    text = adoc.render(_look(), report, _round_coverage(), spec=asp.CONSONANT1)
    assert "`ق→ك`" in text and "`س→ش` ‡ (frozen, G5)" in text


def test_a_frozen_substitution_the_confirm_half_did_not_observe_cannot_certify():
    report = _round_report(_catch())
    report["substitutions"] = {}
    report["g5_frozen"] = ["ق→ك"]
    found = av.verdict(report, spec=asp.CONSONANT1)
    g5 = [condition for condition in found["conditions"] if condition["name"] == "g5.ق→ك"]
    assert g5 and g5[0]["verdict"] == av.CANNOT_CERTIFY


def test_the_noisier_mistake_marks_are_reported_and_never_g6_s():
    groups = cr.descriptive_sites()
    assert set(groups) == set(cr.DESCRIPTIVE_MARKS)
    assert {site.mark for sites in groups.values() for site in sites} <= set(
        cr.DESCRIPTIVE_MARKS)
    assert len(groups["س↔ص"]) == 3 and len(groups["ض↔ظ"]) == 6
    assert groups["ث"] == []  # disclosed, not silently absent
    g6 = set(cr.load_g6_sites())
    assert not any(site.site_id in g6 for sites in groups.values() for site in sites)
    empty = cr.descriptive_catch(("c95",), cache_dir=Path("/nonexistent"))
    assert all(row["read"] == 0 and row["models"] == {} for row in empty.values())


def test_the_round_report_builder_reports_every_observed_substitution(monkeypatch):
    """The builder, not a hand-built report: a substitution outside the freeze is reported and
    never gated (review R1.4)."""
    monkeypatch.setattr(cr, "frozen_substitutions", lambda spec: ("س→ش",))
    items = [_item(0), _item(1)]
    reference = "قَالَ سَططَ"
    decodes = {role: {item.key: {"checks": [{"hop": reference}]} for item in items}
               for role in cr.ROUND_ROLES}
    soft = Flag(("soft_pair",), (), "minor", (), consonants=(("soft_pair", "س→ش"),))
    target = Flag(("target_pair",), (), "minor", (), consonants=(("target_pair", "ق→ك"),))
    flags = {"candidate": (soft, None), "c95": (None, target), "h448": (None, None),
             "base": (None, None)}
    grades = [_grade(role, item, position, flags[role][position])
              for role in cr.ROUND_ROLES for item in items for position in (0, 1)]
    data = cr.RoundData(tarteel=ar.SetData(items, decodes, grades), qdat=None,
                        qud_other=None, qud_half=None, tadabur=None, bench={}, catch=None)
    report = cr.confirm_report(data, spec=asp.CONSONANT1)
    assert set(report["substitutions"]) == {"س→ش", "ق→ك"}
    assert report["g5_frozen"] == ["س→ش"]
    gated = {condition["name"] for condition in report["verdict"]["conditions"]}
    assert "g5.س→ش" in gated and "g5.ق→ك" not in gated


def test_the_report_grades_the_bench_departures_it_found(monkeypatch):
    """A non-empty departure list is grades, not a set: the real-data regeneration found the
    builders reading ``.grades`` off it."""
    item = _item(0)
    reference = "قَالَ سَططَ"
    decodes = {role: {item.key: {"checks": [{"hop": reference}], "stream_b0": reference}}
               for role in cr.ROUND_ROLES}
    grades = [_grade(role, item, position, None)
              for role in cr.ROUND_ROLES for position in (0, 1)]
    data = cr.RoundData(tarteel=None, qdat=ar.SetData([item], decodes, grades),
                        qud_other=None, qud_half=None, tadabur=None,
                        bench={item.audio_filename: reference}, catch=None)
    monkeypatch.setattr("training.nonpro_eval.bench_departures",
                        lambda item, text: {grades[0].word_index})
    report = cr.confirm_report(data, spec=asp.CONSONANT1)
    departures = report["catch"]["qdat_bench_departures"]
    assert departures is not None and departures["comparisons"]


# --- a missing input cannot certify, never crashes (review R1.3) --------------------------------


def test_a_missing_guard_table_cannot_certify_instead_of_crashing():
    report = _round_report(_catch())
    for section, name in (("guards", "qud_other"), ("guards", "tadabur_dev_unseen"),
                          ("main", "all_cause"), ("consistency", "qdat")):
        broken = json.loads(json.dumps(report))
        broken[section][name] = None
        found = av.verdict(broken, spec=asp.CONSONANT1)
        assert all("verdict" in condition for condition in found["conditions"])
        assert found["verdict"] == av.CANNOT_CERTIFY
        reading = {c["name"]: c for c in found["conditions"]}
        assert any(condition.get("reason", "").startswith("no input")
                   for condition in reading.values())


def test_absent_sets_names_the_declared_sets_whose_decodes_are_missing(tmp_path):
    import dataclasses

    spec = dataclasses.replace(asp.CONSONANT1, root=tmp_path, decode_sources={})
    sources = ar.Sources(spec)
    assert ar.absent_sets(sources, "confirm", "ema", spec) == tuple(sorted(
        name for name in asp.CONSONANT1.sets if name in ag.SETS and ag.SETS[name].gated))
    for model in ("candidate_ema", "c95"):
        present = sources.decode_path("qdat", model)
        present.parent.mkdir(parents=True, exist_ok=True)
        present.write_bytes(b"")
    assert "qdat" not in ar.absent_sets(sources, "confirm", "ema", spec)


def test_a_choice_identity_survives_every_absent_corpus_for_the_catch_arm():
    """With no corpus identity record the choice's own candidate identity still binds G6
    (review R2.3): otherwise a supplied cache pair crashes instead of cannot-certifying."""
    identity = {"model_ref": "ref", "weights": "ema", "step": 10_000,
                "recipe": "aug94-mixture-v2", "checkpoint_sha256": "a" * 64}
    choice = {"weights": "ema", "models": {"candidate": "candidate_ema",
                                           "control": "control_ema"},
              "identities": {"candidate_ema": identity, "candidate_live": {}}}
    identities = ag.require_manifest({}, {}, choice, spec=asp.CONSONANT1)
    assert identities == {"candidate_ema": identity}  # the unchosen look arm is not carried
    assert cr._expected_identity(asp.CONSONANT1, "candidate_ema", identities) == identity
    with pytest.raises(ValueError, match="no identity"):
        cr._expected_identity(asp.CONSONANT1, "candidate_ema", {})


def test_a_missing_decodes_root_reads_every_set_absent_not_a_crash(tmp_path):
    """The sitting names a set whose decodes are missing and never fetches it (review R1.3): the
    reviewer's ``Sources.qud_other`` FileNotFoundError must not reach the report."""
    import dataclasses

    spec = dataclasses.replace(asp.CONSONANT1, root=tmp_path, decode_sources={})
    sources = ar.Sources(spec)
    absent = ar.absent_sets(sources, "confirm", "ema", spec)
    assert set(absent) == {"tarteel_v1.confirm", "qdat", "qud_other", "qud_half",
                           "tadabur_dev_unseen"}
    data = cr.load_confirm(sources, "ema", spec, absent=absent)
    assert (data.tarteel, data.qdat, data.qud_other, data.qud_half,
            data.tadabur) == (None,) * 5
    report = cr.confirm_report(data, spec)
    assert report["verdict"]["verdict"] == av.CANNOT_CERTIFY


def test_a_sitting_with_no_decodes_reports_every_set_absent_and_certifies_nothing():
    absent = ("qdat", "qud_half", "qud_other", "tadabur_dev_unseen", "tarteel_v1.confirm")
    data = cr.load_confirm(ar.Sources(asp.CONSONANT1), "ema", asp.CONSONANT1, absent=absent)
    assert (data.tarteel, data.qdat, data.qud_other, data.qud_half,
            data.tadabur) == (None,) * 5
    assert data.bench == {} and data.absent == absent
    report = cr.confirm_report(data, asp.CONSONANT1)
    assert report["absent"] == list(absent)
    assert report["sizes"]["qdat"] is None and report["mechanism"]["qdat"] is None
    assert report["verdict"]["verdict"] == av.CANNOT_CERTIFY
    text = adoc.render(_look(), report, _round_coverage(), spec=asp.CONSONANT1)
    assert "not read (an input is missing); cannot certify" in text
    assert "**Absent inputs:**" in text


def test_the_consonant_doc_renders_c95_and_every_reported_item():
    text = adoc.render(_look(), _round_report(_catch()), _round_coverage(),
                       spec=asp.CONSONANT1)
    assert text.startswith("# The consonant round's readout (#175)")
    assert "| c95 (the baseline) |" in text  # S2.5: the baseline is a column
    assert "| h448 |" in text and "| base (the teacher) |" in text
    assert "where the teacher equals the reference" in text
    assert "G4's companion" in text
    assert "the salted half #95 read" in text
    assert "where the experts heard a departure" in text
    assert "candidate − c95: -1 sites" in text
    assert "Of the 3 held-out mistakes" in text


def test_the_consonant_doc_reads_a_missing_catch_difference_as_not_read():
    catch = dict(_catch(), difference=None)
    text = adoc.render(_look(), _round_report(catch), _round_coverage(),
                       spec=asp.CONSONANT1)
    assert "candidate − c95: not read" in text
    assert "G6 cannot certify" in text


# --- #177: the committed look-half freeze ------------------------------------------------------

#: The G5 list #177 freezes: C95's consonant substitutions behind the most look-half flagged
#: words at the shipped default, the madd alif aside, ties by substitution (Unicode order).
G5_SUBSTITUTIONS = ("ء→∅", "ل→ن", "ن→∅", "ج→ش", "م→∅", "ء→ي", "و→∅", "د→ت", "ر→ل", "ل→∅")


def _committed_freeze() -> dict:
    return json.loads(cr.FREEZE_PATH.read_text(encoding="utf-8"))


def test_the_committed_g5_list_is_ten_consonant_substitutions_without_the_madd_alif():
    rows = _committed_freeze()["substitutions"]
    assert [row["substitution"] for row in rows] == list(G5_SUBSTITUTIONS)
    assert len(rows) == len({row["substitution"] for row in rows}) == 10
    assert {row["cause"] for row in rows} <= set(am.CONSONANT_FLAG_CAUSES)
    assert am.MADD_ALIF not in {row["substitution"] for row in rows}
    assert all(isinstance(row["words"], int) and row["words"] > 0 for row in rows)
    assert rows == sorted(rows, key=lambda row: (-row["words"], row["substitution"]))


def test_the_committed_g5_list_is_c95_s_shipped_tally_s_consonant_top_ten():
    """G5 is read from the **untruncated** shipped-default tally (review R1.1): consonant causes
    only, the madd alif aside. The display list truncates per cause; the tally does not."""
    shipped = _committed_freeze()["models"]["c95"]["scorings"]["shipped_default"]
    full = shipped["substitution_tally"]
    assert {"other_consonant", "soft_pair", "target_pair", "shaddah"} <= set(full)
    for cause, rows in full.items():
        assert all(row["cause"] == cause for row in rows)
        assert rows == sorted(rows, key=lambda row: (-row["words"], row["substitution"]))
        assert shipped["substitutions"][cause] == rows[:10]
    assert len(full["other_consonant"]) > 10  # the tally keeps what the display drops
    rows = [row for cause in am.CONSONANT_FLAG_CAUSES for row in full[cause]
            if row["substitution"] != am.MADD_ALIF]
    assert _committed_freeze()["substitutions"] == sorted(
        rows, key=lambda row: (-row["words"], row["substitution"]))[:10]


def test_the_committed_cause_split_covers_every_word_flags_cause_with_counts_and_rates():
    record = _committed_freeze()
    assert record["half"] == "look"
    assert set(record["models"]) == {"c95", "h448", "base"}
    for model, table in record["models"].items():
        assert set(table["scorings"]) == {"shipped_default", "allowances_off"}
        for scoring, row in table["scorings"].items():
            assert row["graded"] > 0 and row["flagged"] >= 0
            assert row["per_1000"] == round(1000 * row["flagged"] / row["graded"], 2)
            assert set(row["causes"]) == set(CAUSES)
            for cause, entry in row["causes"].items():
                assert 0 <= entry["words"] <= row["flagged"]
                assert entry["per_1000"] == round(1000 * entry["words"] / row["graded"], 2)


def test_the_committed_c95_shipped_row_reproduces_95_s_look_half_totals():
    look = json.loads((ar.READOUT_DIR / "look.json").read_text(encoding="utf-8"))
    committed = look["inputs"]["candidate_ema"]
    row = _committed_freeze()["models"]["c95"]["scorings"]["shipped_default"]
    assert row["graded"] == int(committed["graded_words"])
    assert row["flagged"] == int(committed["flagged_words"])


def test_the_committed_freeze_binds_its_inputs_harness_muraja_and_preregistration():
    record = _committed_freeze()
    assert record["issue"] == 177
    assert "freeze" in record["command"] and "look" in record["command"]
    assert set(record["inputs"]) == {
        "tadabur/aug_readout/decodes/tarteel_v1.look.candidate_ema.jsonl.gz",
        "tadabur/nonpro_eval/tarteel_v1/decodes.h448.jsonl.gz",
        "tadabur/nonpro_eval/tarteel_v1/decodes.base.jsonl.gz"}
    assert not any("confirm" in name for name in record["inputs"])
    for name, digest in record["inputs"].items():
        assert ag.sha256_file(ag.TOOLS_DIR / name) == digest
    assert record["harness"]["binary_sha256"] == asp.HARNESS_SHA256
    assert record["harness"]["muraja_commit"] == asp.MURAJA_COMMIT
    assert record["pins"] == {"muraja_commit": asp.MURAJA_COMMIT,
                              "harness_sha256": asp.HARNESS_SHA256,
                              "prereg_sha256": asp.PREREG_SHA256}
    assert ag.sha256_file(asp.PREREG_PATH) == asp.PREREG_SHA256
    assert record["items"] == 4530


def test_the_freeze_doc_is_generated_from_the_committed_record():
    from training.consonant_freeze_doc import render

    record = _committed_freeze()
    text = cr.FREEZE_DOC_PATH.read_text(encoding="utf-8")
    assert text == render(record)
    assert "C95, look half only" in text


def test_the_freeze_doc_prints_the_top_ten_of_each_cause_s_full_tally():
    """R1.1: the record keeps the whole tally; the generated doc reports each cause's top ten."""
    full = _committed_freeze()["models"]["c95"]["scorings"]["shipped_default"][
        "substitution_tally"]
    text = cr.FREEZE_DOC_PATH.read_text(encoding="utf-8")
    # The same substitution and count can legitimately appear under the other scoring, so the
    # absence check reads the shipped-default block alone (the causes' tables have a header of
    # the same name earlier in the doc).
    substitutions = text.split("## C95's top substitutions by cause", 1)[1]
    shipped = substitutions.split("### Shipped default", 1)[1].split(
        "### Allowances off", 1)[0]
    for rows in full.values():
        for row in rows[:10]:
            assert f"| {row['substitution']} | {row['words']} |" in shipped
        for row in rows[10:]:
            assert f"| {row['substitution']} | {row['words']} |" not in shipped


def test_the_freeze_writer_refuses_the_confirm_half_before_reading_or_writing(tmp_path):
    record, doc = tmp_path / "freeze.json", tmp_path / "freeze.md"
    with pytest.raises(SystemExit, match="look"):
        cr.write_freeze(ConfirmHalf(), half="confirm", path=record, doc_path=doc)
    assert not record.exists() and not doc.exists()


def test_the_freeze_writer_refuses_to_move_a_committed_freeze(tmp_path):
    record, doc = tmp_path / "freeze.json", tmp_path / "freeze.md"
    record.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="exists|fixed|frozen"):
        cr.write_freeze(ConfirmHalf(), half="look", path=record, doc_path=doc)
    assert record.read_text(encoding="utf-8") == "{}\n"
    assert not doc.exists()


def test_the_freeze_doc_reads_g5_s_bound_as_candidate_minus_c95():
    """R1.3: an absolute rate bound would veto an unchanged candidate; the guard is paired."""
    text = cr.FREEZE_DOC_PATH.read_text(encoding="utf-8")
    paragraph = next(line for line in text.splitlines() if "99.5%" in line)
    assert "candidate minus C95" in paragraph


def _tables_for_tally(tally) -> dict:
    plan = [Flag((cause,), (), "minor", (), consonants=((cause, substitution),))
            for substitution, cause, words in tally for _ in range(words)]
    items = [_item(i) for i in range((len(plan) + 1) // 2)]
    grades = [_scored_grade("c95", SHIPPED, item, position, plan[2 * i + position])
              for i, item in enumerate(items) for position in (0, 1)]
    return cr.freeze_tables(grades)


def test_the_freeze_keeps_the_full_tally_beside_each_cause_s_top_ten():
    """R1.1: the record preserves the canonical tally; only the display is truncated."""
    tables = _tables_for_tally(FREEZE_TALLY)
    shipped = tables["models"]["c95"]["scorings"]["shipped_default"]
    expected = sorted(({"cause": cause, "substitution": substitution, "words": words}
                       for substitution, cause, words in FREEZE_TALLY),
                      key=lambda row: (-row["words"], row["substitution"]))
    full = shipped["substitution_tally"]
    for cause in {row["cause"] for row in expected}:
        rows = [row for row in expected if row["cause"] == cause]
        assert full[cause] == rows
        assert shipped["substitutions"][cause] == rows[:10]
    assert len(full["other_consonant"]) == 11


def test_g5_reads_the_untruncated_tally_while_the_diagnostic_is_truncated_per_cause():
    """The madd alif takes a display slot but is not a G5 candidate, so a row eleventh in the
    displayed cause can still be G5's tenth (review R1.1)."""
    ranked = [(f"{letter}→∅", "other_consonant", 19 - i)
              for i, letter in enumerate("رزسشصضطظع")]
    tally = [(am.MADD_ALIF, "other_consonant", 20), *ranked,
             ("ي→∅", "other_consonant", 11)]
    tables = _tables_for_tally(tally)
    assert [row["substitution"] for row in tables["substitutions"]] == [
        *(label for label, _, _ in ranked), "ي→∅"]
    shown = [row["substitution"] for row
             in tables["models"]["c95"]["scorings"]["shipped_default"]["substitutions"]
             ["other_consonant"]]
    assert am.MADD_ALIF in shown and "ي→∅" not in shown


#: A synthetic tally for the freeze's two readings: substitution, cause, flagged words. The
#: diagnostic keeps every cause, the madd alif and the shaddah rows included, and truncates each
#: cause to ten; only G5 excludes them. Ties at 5, 4, 3, 2 and 1 order by substitution only.
FREEZE_TALLY = (
    ("ا→∅", "other_consonant", 99), ("ن→∅", "shaddah", 9), ("ز→س", "other_consonant", 6),
    ("ب→ت", "other_consonant", 5), ("ج→د", "other_consonant", 5),
    ("س→ش", "target_pair", 4), ("ص→س", "soft_pair", 4),
    ("ح→ه", "soft_pair", 3), ("خ→ح", "other_consonant", 3),
    ("د→ض", "other_consonant", 2), ("ر→ل", "other_consonant", 2),
    ("ط→ت", "other_consonant", 2), ("ق→ك", "other_consonant", 2),
    ("ع→ء", "other_consonant", 1), ("ي→ى", "other_consonant", 1),
)


def _scored_grade(model: str, scoring: str, item: ClipItem, position: int,
                  flag: Flag | None) -> WordGrade:
    return WordGrade(model, scoring, item, position, 2, position + 1, 7 + position,
                     SiteGrade("flagged" if flag is not None else "not_flagged"), flag)


def _top(rows: list[dict], cause: str) -> list[dict]:
    """Each cause's top ten from the canonical tally: words first, then substitution."""
    return sorted((row for row in rows if row["cause"] == cause),
                  key=lambda row: (-row["words"], row["substitution"]))[:10]


def test_freeze_tables_takes_the_top_ten_consonant_substitutions_by_words_then_unicode():
    plan = [Flag((cause,), (), "minor", (), consonants=((cause, substitution),))
            for substitution, cause, words in FREEZE_TALLY for _ in range(words)]
    items = [_item(i) for i in range((len(plan) + 1) // 2)]
    grades = [_scored_grade("c95", SHIPPED, item, position, plan[2 * i + position])
              for i, item in enumerate(items) for position in (0, 1)]
    tables = cr.freeze_tables(grades)

    # G5: the round's consonant causes only, the madd alif aside, the list over every cause.
    assert [row["substitution"] for row in tables["substitutions"]] == [
        "ز→س", "ب→ت", "ج→د", "س→ش", "ص→س", "ح→ه", "خ→ح", "د→ض", "ر→ل", "ط→ت"]
    assert len(tables["substitutions"]) == 10

    # The cause diagnostic is the full canonical tally (review R1.1): shaddah and the madd alif
    # stay in, and each cause is truncated to ten only after its own rows are ranked.
    shipped = tables["models"]["c95"]["scorings"]["shipped_default"]
    diagnostic = shipped["substitutions"]
    canonical = flag_view(grades)["substitutions"]
    assert {row["cause"] for row in canonical} == {"other_consonant", "shaddah",
                                                   "soft_pair", "target_pair"}
    for cause in {row["cause"] for row in canonical}:
        assert diagnostic[cause] == _top(canonical, cause)
    assert diagnostic["shaddah"] == [{"cause": "shaddah", "substitution": "ن→∅", "words": 9}]
    assert {"cause": "other_consonant", "substitution": am.MADD_ALIF, "words": 99} \
        in diagnostic["other_consonant"]
    assert len(diagnostic["other_consonant"]) == 10  # the eleventh other_consonant row drops
    assert "ي→ى" not in {row["substitution"] for row in diagnostic["other_consonant"]}
    assert set(shipped["causes"]) == set(CAUSES)


# --- #179: coverage for every set, and RetaSy's absence stated (acceptance criteria) -----------


def test_the_round_discloses_a_coverage_row_for_every_set():
    record = cr.coverage()
    populations = record["populations"]
    assert set(populations) == {
        "tarteel_v1.look", "tarteel_v1.confirm", "qud_other", "qud_half", "qdat",
        "tadabur_dev_unseen", "qdat_bench", "ship_a_verified"}
    for name, row in populations.items():
        assert isinstance(row["decoded"], int) and isinstance(row["evaluated"], int)
        assert row["evaluated"] + row["excluded"] == row["decoded"], name
        assert isinstance(row["excluded_words"], int)
        assert len(row["excluded_keys"]) == row["excluded"]
        assert row["reason"].strip()
    counts = ar.expected_counts(asp.CONSONANT1)
    for name in ("qdat", "tadabur_dev_unseen", "qud_other", "qud_half"):
        assert populations[name]["decoded"] == counts[name], name
    # A set with no reference exclusions says so with zeroes, not a missing count.
    for name in ("qdat", "tadabur_dev_unseen", "qdat_bench", "ship_a_verified"):
        assert populations[name]["excluded"] == 0, name
    assert populations["qdat"]["evaluated"] == 1505
    assert populations["tadabur_dev_unseen"]["evaluated"] == 671
    assert populations["qdat_bench"]["decoded"] == 159  # qdat's clips the bench reads
    assert populations["ship_a_verified"]["decoded"] == len(cr.g6_scope()["sites"]) == 8
    assert populations["tarteel_v1.look"]["evaluated"] == 4530
    assert populations["tarteel_v1.confirm"]["evaluated"] == 4408
    assert populations["qud_half"]["evaluated"] == 10283
    assert populations["qud_other"]["excluded"] == 176  # the pilot's waqf/leen limitation


def test_the_round_s_doc_renders_every_coverage_row():
    record = cr.coverage()
    report = _round_report(_catch())
    sizes = {"tarteel_v1.confirm": "tarteel_v1_confirm", "qdat": "qdat",
             "qud_other": "qud_other", "qud_half": "qud_half",
             "tadabur_dev_unseen": "tadabur_dev_unseen", "qdat_bench": "qdat_bench",
             "ship_a_verified": "ship_a_verified"}
    for population, size in sizes.items():
        report["sizes"][size] = record["populations"][population]["evaluated"]
    look = _look()
    look["items"] = record["populations"]["tarteel_v1.look"]["evaluated"]
    text = adoc.render(look, report, record, spec=asp.CONSONANT1)
    for name in record["populations"]:
        assert f"| {name} |" in text


def test_retasy_is_reported_not_run_with_its_reason():
    data = cr.RoundData(tarteel=None, qdat=None, qud_other=None, qud_half=None,
                        tadabur=None, bench={}, catch=None)
    report = cr.confirm_report(data, spec=asp.CONSONANT1)
    retasy = report["not_run"]["retasy"]
    assert retasy["status"] == "not run"
    assert "#171" in retasy["reason"]
    text = adoc.render(_look(), report, _round_coverage(), spec=asp.CONSONANT1)
    assert "RetaSy" in text and "not run" in text and "#171" in text

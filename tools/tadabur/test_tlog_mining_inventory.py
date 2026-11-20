"""Acceptance tests for the candidate inventory readout (#300, #261 T1).

Committed before the module exists and **locked**: later turns may add tests but never weaken or
rewrite these. The readout is a no-listening count over the committed phase A-C decodes of tlog
confirm.1/2 (5,762 gating + 10,642 mined rows) graded by four detectors; it reads no human
verdict and no look, confirm.3/4 or train row.

The interface, in ``tadabur.tlog_mining.candidate_inventory`` (new)
-------------------------------------------------------------------
Constants
  ``ROOT`` (``tools/tadabur``), ``INVENTORY_DIR`` (``ROOT/tlog_mining/inventory``), ``DOC_PATH``
  (``docs/tlog-mining-inventory.md``); ``DETECTORS = ("base", "c95", "r2_candidate_ema",
  "r3_candidate_ema")``; ``SCORINGS = ("allowances_off", "shipped_default")`` (strict first);
  ``CAUSES == training.word_flags.CAUSES`` (seven, ``unrecorded`` included); ``PINS``: input name
  (a posix path relative to ``ROOT``) -> SHA-256; ``WORD_GRADES_LIMIT_BYTES`` (25 MB).

Inputs (``root`` is ``ROOT`` in production)
  ``decode_name(population, model) -> str`` the input name of a decode file; population is
  ``"gating"`` or ``"mined"``. Gating base/c95/r2 are
  ``consonant2_readout/decodes/tlog.confirm.<model>.jsonl.gz``, gating r3 is
  ``tlog_mining/decodes/tlog.gating.r3_candidate_ema.jsonl.gz``, mined are
  ``tlog_mining/decodes/tlog.mined.<model>.jsonl.gz``.
  ``read_pinned(name, root) -> bytes`` refuses a name not in ``PINS`` (``ValueError``, "not
  pinned") and bytes whose SHA-256 is not the pin ("not the pinned SHA-256").
  ``load_decodes(root) -> {population: {model: {key: checks}}}`` reads every decode through
  ``read_pinned``, keeps each line's ``checks`` only (the r3 gating ``spans`` are ignored), and
  refuses a line whose ``tranche`` is not confirm.1/confirm.2 (the message names "confirm.1") and
  a model whose key set differs from the population's base (the message names the model).
  ``excluded_248(root)`` the 20 #248 clip keys (consonant2 ``report.json``);
  ``listed_unphonetizable(root)`` the 26 keys phase C listed (``meta.r3_candidate_ema.json``);
  ``require_r3_identity(meta)`` refuses a meta whose r3 checkpoint SHA-256 is not
  ``training.aug_readout_spec.R3_CANDIDATE_SHA256``.

Population and grades
  ``clip_table(rows_by_tranche, *, excluded, listed) -> {key: clip}``: the manifest rows of each
  tranche -> one record per clip with ``population`` (``gating`` for the ``gating`` stratum, else
  ``mined``), ``tranche``, ``stratum``, ``label_source``, ``group_key``. Drops ``excluded`` keys
  (gating only); refuses a row in ``listed``.
  ``word_rows(grades, clips) -> list[dict]``: ``training.external_corpus.WordGrade``-shaped grades
  (``model, scoring, item.key, position, word_index, graded, flagged, reason.causes,
  reason.consonants``) paired on ``(key, position)`` into one row per word sorted by
  ``(key, position)``: the clip's fields, ``key``, ``position``, ``word_index`` and, per scoring,
  per detector, ``{"flagged", "causes", "consonants"}``. A word enters only when all four
  detectors graded it under ``allowances_off``.

Counts (rows as ``word_rows`` makes them; JSON-ready, so keys are strings)
  ``population(rows)``, ``counts(rows, scoring)``, ``label_source_trust(rows)``,
  ``catch_loss_context(root)``, ``write_word_grades(rows, directory, *, limit_bytes)``,
  ``build_summary(rows, *, inputs, harness, exclusions, catch_loss, word_grades)``,
  ``render(summary) -> str`` (a ``# `` title, then a plain-language ``## Summary`` section before
  any table, every count of ``summary.json`` after it) and ``write_inventory(rows, *, inputs, harness, exclusions,
  catch_loss, out_dir, doc_path) -> summary``.
"""

from __future__ import annotations

import gzip
import hashlib
import importlib
import json
import subprocess
from collections import Counter
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import pytest

from training import muraja_policy
from training.aug_readout_spec import R3_CANDIDATE_SHA256
from training.nonpro_decode import item_key
from training.word_flags import CAUSES

REPO = Path(__file__).resolve().parents[2]
BASE, C95, R2, R3 = "base", "c95", "r2_candidate_ema", "r3_candidate_ema"
STRICT, SHIPPED = "allowances_off", "shipped_default"

#: input name -> the SHA-256 the committed file has (the pre-registration's §2 for the first three).
EXPECTED_PINS = {
    "consonant2_readout/decodes/tlog.confirm.base.jsonl.gz":
        "91b66205beb25e36044558e91cd96876b6e439c8d33d24ca4165cb896abaf172",
    "consonant2_readout/decodes/tlog.confirm.c95.jsonl.gz":
        "644a1bd1a3e25ceb320ff32e3d3125729e3d3dc14f5270a82627772d0418a5fc",
    "consonant2_readout/decodes/tlog.confirm.r2_candidate_ema.jsonl.gz":
        "4de59c3c8dbceb630a854fb00982b16874885414839c7bc63919c8bc53d1c46a",
    "tlog_mining/decodes/tlog.gating.r3_candidate_ema.jsonl.gz":
        "b6f383bb825502f1e99b006d4eade61bdc26bde90273692f1b2ea7c8d706be80",
    "tlog_mining/decodes/tlog.mined.base.jsonl.gz":
        "a31c4f0535c0bd3a4cd8ff6e7d1b6578fb03e2757e0530a7263a1b6784e49448",
    "tlog_mining/decodes/tlog.mined.c95.jsonl.gz":
        "b9b1e3526a77201a91725943062d3bf4faf15a30bf1b1bc2622321cff0444607",
    "tlog_mining/decodes/tlog.mined.r2_candidate_ema.jsonl.gz":
        "4630d0902c96c884f01357432036522d5d64ebc636f7ca85ffe157f6c9c12150",
    "tlog_mining/decodes/tlog.mined.r3_candidate_ema.jsonl.gz":
        "427e76e7f0c5637ac6c9954ebbbd34f41c5e89081e64b847a7a561c5abdf4066",
    "consonant2_readout/report.json":
        "2149ea0f2f5dc4ba125f718f89f6a942fd2c30a39850f1d259e2823e338b1117",
}
EXTRA_PINNED = ("tlog_mining/decodes/meta.r3_candidate_ema.json",
                "consonant2_readout/catch_loss/result.json",
                "consonant3_readout/catch_loss/result.json")


@pytest.fixture(scope="module")
def ci():
    return importlib.import_module("tadabur.tlog_mining.candidate_inventory")


# --- fixtures: decode files under a temporary root --------------------------------------------


def _gz(path: Path, lines) -> bytes:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = gzip.compress("".join(json.dumps(line, ensure_ascii=False) + "\n"
                                 for line in lines).encode("utf-8"), mtime=0)
    path.write_bytes(data)
    return data


def _line(n: int, tranche: str = "confirm.1", **extra) -> dict:
    return {"key": f"clip{n}.flac#0", "tranche": tranche, "checks": [{"flush": False, "n": n}],
            **extra}


def _decode_root(ci, tmp_path, monkeypatch, *, overrides=None, drop: str | None = None):
    """A root holding all eight decode files for two gating clips and one mined clip, pinned."""
    overrides = overrides or {}
    pins = {}
    for population, keys in (("gating", (1, 2)), ("mined", (3,))):
        for model in ci.DETECTORS:
            name = ci.decode_name(population, model)
            lines = overrides.get((population, model))
            if lines is None:
                lines = [_line(n, **({"spans": "xx"} if (population, model) == ("gating", R3)
                                     else {})) for n in keys]
            data = _gz(tmp_path / name, lines)
            pins[name] = hashlib.sha256(data).hexdigest()
    if drop:
        del pins[drop]
    monkeypatch.setattr(ci, "PINS", pins)
    return tmp_path


# --- pins, joins and refusals -------------------------------------------------------------------


def test_decode_names_are_the_committed_files(ci):
    assert ci.DETECTORS == (BASE, C95, R2, R3)
    assert ci.SCORINGS == (STRICT, SHIPPED)
    assert ci.CAUSES == CAUSES and len(CAUSES) == 7 and "unrecorded" in CAUSES
    for (population, model), name in {
            ("gating", BASE): "consonant2_readout/decodes/tlog.confirm.base.jsonl.gz",
            ("gating", R2): "consonant2_readout/decodes/tlog.confirm.r2_candidate_ema.jsonl.gz",
            ("gating", R3): "tlog_mining/decodes/tlog.gating.r3_candidate_ema.jsonl.gz",
            ("mined", C95): "tlog_mining/decodes/tlog.mined.c95.jsonl.gz"}.items():
        assert ci.decode_name(population, model) == name


def test_every_input_is_pinned_and_the_pins_match_the_committed_files(ci):
    assert ci.ROOT == Path(__file__).resolve().parent
    for name, sha in EXPECTED_PINS.items():
        assert ci.PINS[name] == sha
    for name in EXTRA_PINNED:
        assert name in ci.PINS
    for name, sha in ci.PINS.items():
        assert hashlib.sha256((ci.ROOT / name).read_bytes()).hexdigest() == sha, name


def test_read_pinned_refuses_a_name_that_is_not_pinned(ci, tmp_path, monkeypatch):
    root = _decode_root(ci, tmp_path, monkeypatch)
    (root / "stray.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="not pinned"):
        ci.read_pinned("stray.json", root)
    name = ci.decode_name("mined", BASE)
    assert hashlib.sha256(ci.read_pinned(name, root)).hexdigest() == ci.PINS[name]


def test_read_pinned_refuses_bytes_that_are_not_the_pinned_ones(ci, tmp_path, monkeypatch):
    root = _decode_root(ci, tmp_path, monkeypatch)
    name = ci.decode_name("gating", C95)
    (root / name).write_bytes(b"edited after the pin")
    with pytest.raises(ValueError, match="not the pinned SHA-256"):
        ci.read_pinned(name, root)
    with pytest.raises(ValueError, match="not the pinned SHA-256"):
        ci.load_decodes(root)


def test_load_decodes_refuses_an_unpinned_decode_file(ci, tmp_path, monkeypatch):
    dropped = ci.decode_name("mined", R3)
    root = _decode_root(ci, tmp_path, monkeypatch, drop=dropped)
    with pytest.raises(ValueError, match="not pinned"):
        ci.load_decodes(root)


def test_the_r3_gating_join_is_on_key_and_ignores_spans(ci, tmp_path, monkeypatch):
    root = _decode_root(ci, tmp_path, monkeypatch)
    decodes = ci.load_decodes(root)
    assert set(decodes) == {"gating", "mined"}
    assert set(decodes["gating"]) == set(ci.DETECTORS) == set(decodes["mined"])
    assert decodes["gating"][R3] == {"clip1.flac#0": [{"flush": False, "n": 1}],
                                     "clip2.flac#0": [{"flush": False, "n": 2}]}
    assert decodes["gating"][R3] == decodes["gating"][BASE]
    assert list(decodes["mined"][BASE]) == ["clip3.flac#0"]


def test_a_detector_that_decoded_other_clips_than_the_base_is_refused(ci, tmp_path, monkeypatch):
    root = _decode_root(ci, tmp_path, monkeypatch,
                        overrides={("gating", R3): [_line(1, spans="xx"), _line(9, spans="xx")]})
    with pytest.raises(ValueError, match=R3):
        ci.load_decodes(root)


@pytest.mark.parametrize("tranche", ["look", "confirm.3", "confirm.4", "train", "tlog", None])
def test_a_row_outside_confirm_1_and_2_is_refused(ci, tmp_path, monkeypatch, tranche):
    bad = _line(3)
    bad["tranche"] = tranche
    root = _decode_root(ci, tmp_path, monkeypatch, overrides={("mined", C95): [bad]})
    with pytest.raises(ValueError, match="confirm.1"):
        ci.load_decodes(root)


def test_the_r3_identity_must_be_the_pinned_checkpoint(ci):
    meta = json.loads((ci.ROOT / "tlog_mining/decodes/meta.r3_candidate_ema.json")
                      .read_text(encoding="utf-8"))
    ci.require_r3_identity(meta)
    identity = meta["identities"]["tlog.mined/r3_candidate_ema"]["identity"]
    assert identity["checkpoint_sha256"] == R3_CANDIDATE_SHA256
    identity["checkpoint_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="checkpoint"):
        ci.require_r3_identity(meta)
    with pytest.raises(ValueError, match="r3_candidate_ema"):
        ci.require_r3_identity({"identities": {}})


# --- the real committed decodes: population, #248 and the 26 listed rows ------------------------


@pytest.fixture(scope="module")
def real(ci):
    manifests = {}
    for tranche in ("confirm.1", "confirm.2"):
        path = ci.ROOT / "tlog" / "tranches" / f"{tranche}.jsonl"
        manifests[tranche] = [json.loads(line) for line in path.read_text(encoding="utf-8")
                              .splitlines()]
    return SimpleNamespace(decodes=ci.load_decodes(ci.ROOT), manifests=manifests,
                           excluded=ci.excluded_248(ci.ROOT),
                           listed=ci.listed_unphonetizable(ci.ROOT))


def test_the_real_decodes_hold_the_rows_the_issue_counts(real):
    assert {m: len(real.decodes["gating"][m]) for m in real.decodes["gating"]} == {
        BASE: 5762, C95: 5762, R2: 5762, R3: 5762}
    assert {m: len(real.decodes["mined"][m]) for m in real.decodes["mined"]} == {
        BASE: 10642, C95: 10642, R2: 10642, R3: 10642}
    assert not set(real.decodes["gating"][BASE]) & set(real.decodes["mined"][BASE])


def test_the_20_excluded_and_the_26_listed_rows_are_where_the_issue_says(real):
    gating, mined = set(real.decodes["gating"][BASE]), set(real.decodes["mined"][BASE])
    assert len(real.excluded) == 20 and real.excluded <= gating
    assert len(real.listed) == 26 and not real.listed & (gating | mined)
    in_manifest = {item_key(r["audio_path"]) for rows in real.manifests.values() for r in rows}
    assert real.listed <= in_manifest


def _rows_of_decoded_clips(real) -> dict[str, list[dict]]:
    decoded = set(real.decodes["gating"][BASE]) | set(real.decodes["mined"][BASE])
    return {tranche: [r for r in rows if item_key(r["audio_path"]) in decoded]
            for tranche, rows in real.manifests.items()}


def test_the_real_clip_table_is_5742_gating_clips_in_5657_group_keys_and_10642_mined(ci, real):
    clips = ci.clip_table(_rows_of_decoded_clips(real), excluded=real.excluded,
                          listed=real.listed)
    gating = [c for c in clips.values() if c["population"] == "gating"]
    assert len(gating) == 5742 and len({c["group_key"] for c in gating}) == 5657
    assert not real.excluded & set(clips)
    assert Counter(c["population"] for c in clips.values()) == {"gating": 5742, "mined": 10642}
    assert {c["tranche"] for c in clips.values()} == {"confirm.1", "confirm.2"}
    assert {c["label_source"] for c in gating} == {"both"}
    assert {c["label_source"] for c in clips.values() if c["population"] == "mined"} == {
        "both", "filename", "identified"}


def _manifest_row(n: int, stratum: str = "gating", label_source: str = "both") -> dict:
    return {"audio_path": f"clip{n}.flac", "group_key": f"g{n}", "stratum": stratum,
            "label_source": label_source, "assigned_label": "1:1", "filename_label": "1:1"}


def test_clip_table_drops_excluded_gating_clips_and_refuses_a_listed_row(ci):
    rows = {"confirm.1": [_manifest_row(1), _manifest_row(2),
                          _manifest_row(3, "reported:straddles", "identified")],
            "confirm.2": [_manifest_row(4, "reported:below_bar", "filename")]}
    clips = ci.clip_table(rows, excluded={"clip2.flac#0"}, listed=set())
    assert sorted(clips) == ["clip1.flac#0", "clip3.flac#0", "clip4.flac#0"]
    assert clips["clip1.flac#0"] == {**clips["clip1.flac#0"], "population": "gating",
                                     "tranche": "confirm.1", "stratum": "gating",
                                     "label_source": "both", "group_key": "g1"}
    assert clips["clip3.flac#0"]["population"] == "mined"
    assert clips["clip4.flac#0"]["tranche"] == "confirm.2"
    with pytest.raises(ValueError, match="listed"):
        ci.clip_table(rows, excluded=set(), listed={"clip3.flac#0"})


def test_clip_table_refuses_a_row_of_another_tranche(ci):
    with pytest.raises(ValueError, match="confirm.1"):
        ci.clip_table({"look": [_manifest_row(1)]}, excluded=set(), listed=set())
    with pytest.raises(ValueError, match="confirm.1"):
        ci.clip_table({"confirm.3": [_manifest_row(1)]}, excluded=set(), listed=set())


# --- grades -> word rows --------------------------------------------------------------------------


def _grade(model, scoring, key, position, *, flagged=False, causes=(), consonants=(),
           word_index=None, graded=True):
    reason = SimpleNamespace(causes=tuple(causes), consonants=tuple(map(tuple, consonants))) \
        if flagged else None
    return SimpleNamespace(model=model, scoring=scoring, item=SimpleNamespace(key=key),
                           position=position,
                           word_index=position if word_index is None else word_index,
                           graded=graded, flagged=flagged, reason=reason)


CLIP = {"population": "mined", "tranche": "confirm.2", "stratum": "reported:straddles",
        "label_source": "identified", "group_key": "g1"}


def _all_models(key, position, **kw):
    return [_grade(m, s, key, position, **kw)
            for m in (BASE, C95, R2, R3) for s in (STRICT, SHIPPED)]


def test_word_rows_pair_the_four_detectors_on_key_and_position(ci):
    grades = (_all_models("a#0", 1)
              + [_grade(m, s, "a#0", 0, flagged=m == C95 or s == STRICT and m == R3,
                        causes=("soft_pair",), consonants=[("soft_pair", "س→ص")])
                 for m in ci.DETECTORS for s in ci.SCORINGS])
    (first, second) = ci.word_rows(grades, {"a#0": CLIP})
    assert (first["key"], first["position"], second["position"]) == ("a#0", 0, 1)
    assert {k: first[k] for k in CLIP} == CLIP
    assert first[STRICT][C95] == {"flagged": True, "causes": ["soft_pair"],
                                  "consonants": [["soft_pair", "س→ص"]]}
    assert first[STRICT][BASE] == {"flagged": False, "causes": [], "consonants": []}
    assert first[SHIPPED][R3]["flagged"] is False and first[SHIPPED][C95]["flagged"] is True
    assert set(first[STRICT]) == set(first[SHIPPED]) == set(ci.DETECTORS)
    assert second[STRICT][C95]["flagged"] is False


def test_a_word_one_detector_did_not_grade_is_not_in_the_population(ci):
    grades = _all_models("a#0", 0) + [
        _grade(m, s, "a#0", 1, graded=not (m == R3 and s == STRICT))
        for m in ci.DETECTORS for s in ci.SCORINGS]
    rows = ci.word_rows(grades, {"a#0": CLIP})
    assert [r["position"] for r in rows] == [0]


def test_detectors_that_graded_different_words_are_refused(ci):
    grades = _all_models("a#0", 0)
    grades[2] = _grade(C95, STRICT, "a#0", 0, word_index=5)
    with pytest.raises(ValueError, match="word"):
        ci.word_rows(grades, {"a#0": CLIP})


def test_a_grade_of_a_clip_outside_the_table_is_refused(ci):
    with pytest.raises(ValueError, match="clip"):
        ci.word_rows(_all_models("excluded#0", 0), {"a#0": CLIP})


# --- the hand-counted fixture --------------------------------------------------------------------


def _flag(causes=(), pairs=()):
    return {"flagged": True, "causes": list(causes), "consonants": [[c, p] for c, p in pairs]}


OFF = {"flagged": False, "causes": [], "consonants": []}


def _word(key, position, group, flags, *, population="mined", tranche="confirm.2",
          stratum="reported:straddles", label_source="identified", shipped=None) -> dict:
    strict = {m: flags.get(m, OFF) for m in (BASE, C95, R2, R3)}
    return {"key": key, "position": position, "word_index": position, "population": population,
            "tranche": tranche, "stratum": stratum, "label_source": label_source,
            "group_key": group, STRICT: strict,
            SHIPPED: strict if shipped is None else {m: shipped.get(m, OFF) for m in strict}}


NL = ("other_consonant", "ن→ل")
SS = ("soft_pair", "س→ص")
SH = ("target_pair", "س→ش")
GATING = {"population": "gating", "tranche": "confirm.1", "stratum": "gating",
          "label_source": "both"}


def fixture_rows() -> list[dict]:
    return [
        _word("k1#0", 0, "g1", {}, **GATING),                                       # W1  none
        _word("k1#0", 1, "g1", {BASE: _flag(["other_consonant"], [NL]),            # W2  all four
                                C95: _flag(["other_consonant"], [NL]),
                                R2: _flag(["other_consonant"], [NL]),
                                R3: _flag(["soft_pair"], [SS])}, **GATING),
        _word("k2#0", 0, "g2", {BASE: _flag(["tashkeel"])}, **GATING),              # W3  base
        _word("k2#0", 1, "g2", {R3: _flag(["shaddah"])}, **GATING),                 # W4  r3
        _word("k3#0", 0, "g3", {C95: _flag(["other_consonant"], [NL]),             # W5  c95+r2
                                R2: _flag(["other_consonant"], [NL])}),
        _word("k3#0", 1, "g3", {BASE: _flag(["skipped"]),                           # W6  b+c+r2
                                C95: _flag(["skipped", "unrecorded"]),
                                R2: _flag(["skipped"])}),
        _word("k4#0", 0, "g4", {R2: _flag(["target_pair"], [SH]),                   # W7  r2+r3
                                R3: _flag(["target_pair"], [SH])},
              stratum="reported:below_bar", label_source="filename", shipped={}),
        _word("k4#0", 1, "g4", {}, stratum="reported:below_bar", label_source="filename"),  # W8
        _word("k5#0", 0, "g5", {}, stratum="reported:identified"),                  # W9  none
    ]


def test_population_counts_clips_group_keys_and_words(ci):
    assert ci.population(fixture_rows()) == {
        "gating": {"clips": 2, "group_keys": 2, "words": 4},
        "mined": {"clips": 3, "group_keys": 3, "words": 5},
        "all": {"clips": 5, "group_keys": 5, "words": 9}}


def test_agreement_levels_and_all_sixteen_patterns_are_counted(ci):
    strict = ci.counts(fixture_rows(), STRICT)
    assert strict["words"] == 9
    assert strict["by_level"] == {"0": 3, "1": 2, "2": 2, "3": 1, "4": 1}
    patterns = strict["by_pattern"]
    assert len(patterns) == 16
    assert set(patterns) == {"none"} | {"+".join(c) for n in range(1, 5)
                                         for c in combinations(ci.DETECTORS, n)}
    assert {name: n for name, n in patterns.items() if n} == {
        "none": 3, BASE: 1, R3: 1, f"{C95}+{R2}": 1, f"{R2}+{R3}": 1, f"{BASE}+{C95}+{R2}": 1,
        "+".join(ci.DETECTORS): 1}


def test_causes_are_counted_per_word_over_every_detector_that_flags_it(ci):
    by_cause = ci.counts(fixture_rows(), STRICT)["by_cause"]
    assert by_cause == {"tashkeel": 1, "soft_pair": 1, "target_pair": 1, "other_consonant": 2,
                        "shaddah": 1, "skipped": 1, "unrecorded": 1}
    assert set(by_cause) == set(CAUSES)


def test_substitution_pairs_count_words_most_frequent_first_then_by_pair(ci):
    assert ci.counts(fixture_rows(), STRICT)["substitutions"][:3] == [
        {"pair": "ن→ل", "words": 2}, {"pair": "س→ش", "words": 1}, {"pair": "س→ص", "words": 1}]


def test_tranche_stratum_and_label_source_counts(ci):
    strict = ci.counts(fixture_rows(), STRICT)

    def pick(table):
        return {name: (entry["words"], entry["flagged_words"]) for name, entry in table.items()}

    assert pick(strict["by_tranche"]) == {"confirm.1": (4, 3), "confirm.2": (5, 3)}
    assert pick(strict["by_stratum"]) == {
        "gating": (4, 3), "reported:straddles": (2, 2), "reported:below_bar": (2, 1),
        "reported:identified": (1, 0)}
    assert pick(strict["by_label_source"]) == {
        "gating:both": (4, 3), "mined:identified": (3, 2), "mined:filename": (2, 1)}


def test_concentration_is_flagged_words_per_group_key(ci):
    concentration = ci.counts(fixture_rows(), STRICT)["concentration"]
    assert (concentration["flagged_words"], concentration["group_keys"]) == (6, 4)
    assert concentration["top"][:4] == [
        {"group_key": "g2", "flagged_words": 2}, {"group_key": "g3", "flagged_words": 2},
        {"group_key": "g1", "flagged_words": 1}, {"group_key": "g4", "flagged_words": 1}]


def test_shipped_default_is_counted_beside_the_strict_scoring_and_differs_from_it(ci):
    shipped = ci.counts(fixture_rows(), SHIPPED)
    assert shipped["by_level"] == {"0": 4, "1": 2, "2": 1, "3": 1, "4": 1}
    assert shipped["by_label_source"]["mined:filename"]["flagged_words"] == 0
    assert ci.counts(fixture_rows(), STRICT)["by_level"]["0"] == 3


# --- reference trust ------------------------------------------------------------------------------


def test_flag_rate_per_label_source_against_the_gating_baseline(ci):
    trust = ci.label_source_trust(fixture_rows())
    assert set(trust) == {"gating", "identified", "filename"}
    assert (trust["gating"]["words"], trust["gating"]["flagged_words"]) == (4, 3)
    assert trust["gating"]["rate"] == 0.75
    assert (trust["identified"]["words"], trust["identified"]["flagged_words"]) == (3, 2)
    assert trust["identified"]["rate"] == pytest.approx(2 / 3)
    assert trust["identified"]["difference_from_gating"] == pytest.approx(2 / 3 - 0.75)
    for name, entry in trust.items():
        low, high = entry["ci95"]
        assert low <= entry["rate"] <= high, name
    assert "difference_from_gating" not in trust["gating"]
    assert len(trust["filename"]["difference_ci95"]) == 2


def _trust_rows(per_group_flags: list[list[bool]], population="gating", label="both"):
    rows = []
    for g, flags in enumerate(per_group_flags):
        for i, flagged in enumerate(flags):
            rows.append(_word(f"{population}{g}#0", i, f"{population}{g}",
                              {BASE: _flag(["other_consonant"])} if flagged else {},
                              population=population, label_source=label,
                              stratum="gating" if population == "gating" else "reported:identified"))
    return rows


def test_the_interval_resamples_group_keys_not_words(ci):
    even = ci.label_source_trust(_trust_rows([[True, False]] * 4))
    assert even["gating"]["rate"] == 0.5 and even["gating"]["ci95"] == [0.5, 0.5]
    lopsided = ci.label_source_trust(_trust_rows([[True] * 3, [False] * 3]))
    assert lopsided["gating"]["ci95"] == [0.0, 1.0]


def test_the_interval_is_deterministic(ci):
    rows = _trust_rows([[True, False, False], [False] * 3, [True, True, False], [False] * 2]) \
        + _trust_rows([[True], [False, False], [True, False]], "mined", "filename")
    assert ci.label_source_trust(rows) == ci.label_source_trust(list(rows))


# --- the catch-loss context -------------------------------------------------------------------------


def test_catch_loss_context_is_the_committed_audits_real_shares_by_cause(ci):
    context = ci.catch_loss_context(ci.ROOT)
    assert set(context) == {"round2", "round3"}
    assert context["round2"]["real_total"] == 11 and context["round3"]["real_total"] == 22
    assert {c: (v["real"], v["decisive"]) for c, v in context["round2"]["by_cause"].items()} == {
        "other_consonant": (9, 119), "skipped": (0, 7), "soft_pair": (2, 3),
        "tashkeel": (1, 15), "unrecorded": (0, 1), "shaddah": (0, 1), "target_pair": (0, 1)}
    assert {c: (v["real"], v["decisive"]) for c, v in context["round3"]["by_cause"].items()} == {
        "other_consonant": (18, 114), "shaddah": (0, 3), "skipped": (0, 2),
        "soft_pair": (1, 6), "target_pair": (1, 2), "tashkeel": (3, 17), "unrecorded": (0, 1)}


# --- word_grades.jsonl.gz and its 25 MB fallback --------------------------------------------------------


def _read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def test_the_word_grades_file_holds_every_word_with_both_scorings(ci, tmp_path):
    assert 25_000_000 <= ci.WORD_GRADES_LIMIT_BYTES <= 25 * 2**20
    rows = sorted(fixture_rows(), key=lambda r: (r["key"], r["position"]))
    report = ci.write_word_grades(rows, tmp_path, limit_bytes=ci.WORD_GRADES_LIMIT_BYTES)
    path = tmp_path / "word_grades.jsonl.gz"
    assert report["file"] == "word_grades.jsonl.gz" and report["complete"] is True
    assert report["bytes"] == path.stat().st_size
    assert _read_gz(path) == rows
    assert STRICT in _read_gz(path)[0] and SHIPPED in _read_gz(path)[0]


def test_over_the_limit_only_flagged_words_and_per_clip_word_counts_are_kept(ci, tmp_path):
    rows = [_word("k1#0", 0, "g1", {}), _word("k1#0", 1, "g1", {BASE: _flag(["tashkeel"])}),
            _word("k1#0", 2, "g1", {}, shipped={C95: _flag(["tashkeel"])}),
            _word("k2#0", 0, "g2", {})]
    report = ci.write_word_grades(rows, tmp_path, limit_bytes=1)
    assert report["complete"] is False and report["bytes"] > 1
    assert report["bytes"] == (tmp_path / "word_grades.jsonl.gz").stat().st_size
    kept = _read_gz(tmp_path / "word_grades.jsonl.gz")
    assert [(r["key"], r["position"]) for r in kept] == [("k1#0", 1), ("k1#0", 2)]
    assert report["clip_word_counts_file"] == "clip_word_counts.jsonl.gz"
    assert _read_gz(tmp_path / "clip_word_counts.jsonl.gz") == [
        {"key": "k1#0", "words": 3}, {"key": "k2#0", "words": 1}]


def test_the_complete_file_leaves_no_clip_count_file_behind(ci, tmp_path):
    ci.write_word_grades(fixture_rows(), tmp_path, limit_bytes=ci.WORD_GRADES_LIMIT_BYTES)
    assert not (tmp_path / "clip_word_counts.jsonl.gz").exists()


# --- summary.json and the generated doc ---------------------------------------------------------------------

HARNESS = {"build": {"muraja_commit": "c" * 40}, "muraja_commit": "c" * 40}
INPUTS = {"a/b.gz": "a" * 64}
EXCLUSIONS = {"gating_decoded": 4, "gating_excluded_248": 0, "mined_decoded": 5,
              "listed_unphonetizable": 0}
CATCH_LOSS = {"round2": {"real_total": 11, "by_cause": {"tashkeel": {"real": 1, "decisive": 15}}},
              "round3": {"real_total": 22, "by_cause": {"tashkeel": {"real": 3, "decisive": 17}}}}
SUMMARY_KEYS = {"inputs", "harness", "population", "exclusions", "strict", "shipped_default",
                "label_source_trust", "catch_loss_context", "word_grades"}


def _summary(ci, rows=None):
    rows = fixture_rows() if rows is None else rows
    return ci.build_summary(rows, inputs=INPUTS, harness=HARNESS, exclusions=EXCLUSIONS,
                            catch_loss=CATCH_LOSS,
                            word_grades={"file": "word_grades.jsonl.gz", "bytes": 9,
                                         "complete": True})


def test_the_summary_holds_every_section_the_issue_lists(ci):
    summary = _summary(ci)
    assert set(summary) >= SUMMARY_KEYS
    assert summary["inputs"] == INPUTS and summary["harness"] == HARNESS
    assert summary["exclusions"] == EXCLUSIONS and summary["catch_loss_context"] == CATCH_LOSS
    assert summary["population"] == ci.population(fixture_rows())
    assert summary["strict"] == ci.counts(fixture_rows(), STRICT)
    assert summary["shipped_default"] == ci.counts(fixture_rows(), SHIPPED)
    assert summary["label_source_trust"] == ci.label_source_trust(fixture_rows())
    assert summary["word_grades"]["complete"] is True
    json.dumps(summary, ensure_ascii=False)


def test_the_summary_has_no_per_detector_catch_precision_or_weights(ci):
    text = json.dumps(_summary(ci), ensure_ascii=False).lower()
    for dropped in ("precision", "catch_rate", "horvitz", "inclusion_probability"):
        assert dropped not in text


def test_the_doc_is_rendered_from_the_summary_with_a_plain_summary_first(ci):
    text = ci.render(_summary(ci))
    lines = text.splitlines()
    assert lines[0].startswith("# ")
    assert "summary.json" in text
    heading = next(i for i, line in enumerate(lines) if line.startswith("## Summary"))
    first_table = next(i for i, line in enumerate(lines) if line.startswith("|"))
    assert heading < first_table
    assert "9" in text and "ن→ل" in text
    for pattern in _summary(ci)["strict"]["by_pattern"]:
        assert pattern in text
    assert ci.render(_summary(ci)) == text


def _write(ci, root: Path):
    return ci.write_inventory(fixture_rows(), inputs=INPUTS, harness=HARNESS,
                              exclusions=EXCLUSIONS, catch_loss=CATCH_LOSS,
                              out_dir=root / "inventory", doc_path=root / "doc.md")


def test_write_inventory_writes_the_three_artifacts_and_the_doc_matches_the_summary(ci, tmp_path):
    summary = _write(ci, tmp_path)
    written = json.loads((tmp_path / "inventory" / "summary.json").read_text(encoding="utf-8"))
    assert written == summary
    assert "ن→ل" in (tmp_path / "inventory" / "summary.json").read_text(encoding="utf-8")
    assert (tmp_path / "doc.md").read_text(encoding="utf-8") == ci.render(written)
    path = tmp_path / "inventory" / "word_grades.jsonl.gz"
    assert written["word_grades"]["bytes"] == path.stat().st_size


def test_a_rerun_is_byte_identical(ci, tmp_path):
    _write(ci, tmp_path / "one")
    _write(ci, tmp_path / "two")
    for relative in ("inventory/summary.json", "inventory/word_grades.jsonl.gz", "doc.md"):
        assert (tmp_path / "one" / relative).read_bytes() == (tmp_path / "two" / relative).read_bytes()


# --- the committed artifacts --------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def committed(ci):
    summary = json.loads((ci.INVENTORY_DIR / "summary.json").read_text(encoding="utf-8"))
    return SimpleNamespace(summary=summary)


def test_the_committed_summary_pins_the_inputs_and_the_harness(ci, committed):
    summary = committed.summary
    assert set(summary) >= SUMMARY_KEYS
    detector_inputs = {ci.decode_name(p, m) for p in ("gating", "mined") for m in ci.DETECTORS}
    assert detector_inputs <= set(summary["inputs"])
    for name, sha in summary["inputs"].items():
        assert ci.PINS[name] == sha, name
    assert summary["harness"]["muraja_commit"] == muraja_policy.MURAJA_COMMIT


def test_the_committed_summary_counts_the_issue_s_population(committed):
    summary = committed.summary
    assert summary["exclusions"] == {"gating_decoded": 5762, "gating_excluded_248": 20,
                                     "mined_decoded": 10642, "listed_unphonetizable": 26}
    assert summary["population"]["gating"]["clips"] == 5742
    assert summary["population"]["gating"]["group_keys"] == 5657
    assert summary["population"]["mined"]["clips"] <= 10642
    total = summary["population"]["all"]["words"]
    assert total == summary["strict"]["words"] == summary["shipped_default"]["words"]
    assert sum(summary["strict"]["by_pattern"].values()) == total
    assert set(summary["label_source_trust"]) == {"gating", "both", "filename", "identified"}


def test_the_committed_word_grades_match_the_summary(ci, committed):
    report = committed.summary["word_grades"]
    path = ci.INVENTORY_DIR / report["file"]
    assert path.stat().st_size == report["bytes"]
    rows = _read_gz(path)
    if report["complete"]:
        assert len(rows) == committed.summary["population"]["all"]["words"]
        assert report["bytes"] <= ci.WORD_GRADES_LIMIT_BYTES
        assert set(rows[0]) >= {STRICT, SHIPPED, "key", "position"}
    else:
        counts = _read_gz(ci.INVENTORY_DIR / report["clip_word_counts_file"])
        assert sum(c["words"] for c in counts) == committed.summary["population"]["all"]["words"]
        assert all(any(r[s][m]["flagged"] for s in ci.SCORINGS for m in ci.DETECTORS)
                   for r in rows)


def test_the_committed_doc_is_the_summary_rendered(ci, committed):
    assert ci.DOC_PATH == REPO / "docs" / "tlog-mining-inventory.md"
    assert ci.DOC_PATH.read_text(encoding="utf-8") == ci.render(committed.summary)


def test_no_inventory_artifact_is_git_ignored(ci):
    paths = [p for p in ci.INVENTORY_DIR.iterdir() if p.is_file()] + [ci.DOC_PATH]
    assert paths
    for path in paths:
        ignored = subprocess.run(["git", "check-ignore", "-q", str(path)], cwd=REPO)
        assert ignored.returncode == 1, f"{path} is git-ignored"

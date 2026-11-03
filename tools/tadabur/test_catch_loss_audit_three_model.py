"""The round-3 catch-loss audit's three-model draw (#257): ``tadabur.catch_loss_audit.draw`` on
``CONSONANT3``.

Round 3 audits only the *new* removals, so the draw compares three models on the look gating
words: C95, the round-2 EMA (the spec's ``r2`` reference) and the chosen round-3 weights.

* **removed** — C95 flags, ``r2`` flags and the chosen round-3 model does not (drawn 100);
* **added** — the round-3 model flags and neither C95 nor ``r2`` does (drawn 50);
* a word one of the three did not grade, and every other combination (above all a word C95
  flags and ``r2`` does not: round 2 already audited it), is in neither stratum.

The look sitting's committed export (``look_grades.jsonl.gz``) carries C95 and the chosen
candidate and is never rewritten. The ``r2`` grades come from a second committed file,
``look_grades_r2.jsonl.gz``, with the same row schema: the ``r2`` rows and a regenerated copy of
C95's. The draw refuses unless that file is committed, holds ``r2`` rows, repeats C95's grades
word for word, and (when the spec declares ``catch_loss_disclosure``, the pre-registration's
disclosed counts) reproduces them: words C95 flags, words ``r2`` flags and words both flag.

The sitting is real, as in :mod:`tadabur.test_catch_loss_audit_draw`: ``run_look`` on synthetic
grades in a throwaway git repository, under the real ``CONSONANT3`` spec. Items sit in blocks
(``n`` is the item index; the first word is graded, the second is flagged by nobody):

====  ==========  ==========  ==========  ==========================================
block  C95         r2          r3          stratum
====  ==========  ==========  ==========  ==========================================
A      flags       flags       -           removed
B      flags       -           -           neither (round 2's removal)
C      flags       -           flags       neither
D      -           -           flags       added
E      -           flags       flags       neither
F      -           flags       -           neither
G      flags       flags       flags       neither
====  ==========  ==========  ==========  ==========================================

The last section reads the committed real files: the regenerated ``look_grades_r2.jsonl.gz``
against the prereg's disclosed counts, and the drawn set once it is committed.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tadabur import catch_loss_audit as audit
from tadabur.test_catch_loss_audit_draw import (
    LookAudit as Round2Audit, WORDS, _Sources, _head, _item, _jsonl, _origin, _words)
from tadabur.tlog_pass1 import read_jsonl_gz
from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import aug_readout_spec as asp
from training import round2_fixtures as fx

REFUSED = (SystemExit, ValueError)
SEED = "round3-catch-loss-2026-11"
EXPORT = ar.LOOK_GRADES
REFERENCE_EXPORT = "look_grades_r2.jsonl.gz"
REFERENCE = "r2"
B, C, E, F, G = 25, 40, 15, 15, 10  # block sizes; A (removed) and D (added) are the parameters
R3_MODELS = ("r3_candidate_live", "r3_candidate_ema")
R2_FLAG = fx.consonant_flag("ق→ك")


def _c95_flag(n: int):
    return (fx.TASHKEEL_FLAG, fx.consonant_flag("س→ص"), fx.consonant_flag("س→ش"))[n % 3]


def _r3_flag(n: int):
    return fx.TASHKEEL_FLAG if n % 2 == 0 else fx.consonant_flag("ث→س")


class Round3Audit:
    """A committed round-3 look sitting plus the ``r2`` grades file, in blocks A to G above.
    ``edit`` rewrites the look export's rows and ``edit_reference`` the ``r2`` file's rows
    before their commit; ``uncommitted`` names the files left out of it."""

    def __init__(self, root, *, removed=130, added=60, edit=None, edit_reference=None,
                 uncommitted=(), disclosure=None):
        self.repo = fx.init_repo(root / "repo")
        self.prereg = fx.commit_prereg(self.repo)
        spec = dataclasses.replace(
            asp.CONSONANT3, root=self.repo / "readout", prereg_path=self.prereg[0],
            pins=dataclasses.replace(asp.CONSONANT3.pins, prereg_sha256=self.prereg[1]))
        self.layout = ag.Layout(spec.root)
        self.layout.root.mkdir()
        records = {f"tlog.look/{m}": fx.model_record(spec, m) for m in (*R3_MODELS, asp.C95)}
        records["tlog.look/base"] = {
            "model": "base", "fingerprints": ag.expected_fingerprints(fx.DEFAULT_MODELS["base"]),
            "identity": ag.baseline_identities()["base"]}
        self.layout.models.write_text(json.dumps(records), encoding="utf-8")
        a = removed
        self.blocks = {"A": range(0, a), "B": range(a, a + B), "C": range(a + B, a + B + C)}
        start = a + B + C
        self.blocks["D"] = range(start, start + added)
        start += added
        for name, size in (("E", E), ("F", F), ("G", G)):
            self.blocks[name] = range(start, start + size)
            start += size
        items = [_item(n) for n in range(start)]
        self.items = items

        def has(name, n):
            return n in self.blocks[name]

        c95_flags = lambda n: any(has(k, n) for k in "ABCG")
        r2_flags = lambda n: any(has(k, n) for k in "AEFG")
        r3_flags = lambda n: any(has(k, n) for k in "CDEG")

        def flags(model, n, position):
            if position == 1:
                return None
            if model == asp.C95:
                return _c95_flag(n) if c95_flags(n) else None
            return _r3_flag(n) if r3_flags(n) else None

        self.c95_flags = sum(c95_flags(n) for n in range(start))
        self.r2_flags = sum(r2_flags(n) for n in range(start))
        self.both_flags = sum(c95_flags(n) and r2_flags(n) for n in range(start))
        # the fixture's hand counts are not the pre-registration's: no disclosure unless given
        self.spec = dataclasses.replace(spec, catch_loss_disclosure=disclosure or {})
        ar.run_look(_Sources(root / "files", items, flags, spec=spec), self.layout, spec=spec)
        self.export = self.layout.root / EXPORT
        rows = _read(self.export)
        if edit is not None:
            ar.write_look_grades(self.export, edit(rows))
        c95_rows = [row for row in rows if row["model"] == asp.C95]
        reference_rows = [dict(row) for row in c95_rows]
        index = {item.key: n for n, item in enumerate(items)}
        for row in c95_rows:
            n = index[row["key"]]
            flagged = row["position"] == 0 and r2_flags(n)
            reference_rows.append({
                **row, "model": REFERENCE, "flagged": flagged,
                "outcome": "flagged" if flagged else "not_flagged",
                "causes": list(R2_FLAG.causes) if flagged else [],
                "consonants": [list(pair) for pair in R2_FLAG.consonants] if flagged else []})
        reference_rows.sort(key=lambda row: (row["model"], row["key"], row["position"]))
        self.reference_export = self.layout.root / REFERENCE_EXPORT
        ar.write_look_grades(self.reference_export, (edit_reference or (lambda r: r))(reference_rows))
        names = [self.layout.look.name, self.layout.choice.name, self.layout.claim("look").name,
                 self.export.name, self.reference_export.name]
        fx.git(self.layout.root, "add", *(n for n in names if n not in uncommitted))
        fx.git(self.layout.root, "commit", "-qm", "the look")
        self.tranche = root / "look.jsonl"
        self.tranche.write_text("".join(json.dumps({
            "audio_path": item.audio_filename, "audio_sha256": item.audio_sha256,
            "split": "clean", "shard_file": f"clean-{n // 10:05d}", "row": n % 10,
            "stratum": "gating"}) + "\n" for n, item in enumerate(items)), encoding="utf-8")
        self.out = self.spec.root / "catch_loss"

    @property
    def disclosed(self) -> dict:
        return {"c95": self.c95_flags, "reference": self.r2_flags, "both": self.both_flags}

    def draw(self, spec=None) -> dict:
        return audit.draw(spec or self.spec, uthmani_words=_words, tranche=self.tranche)

    def worklist(self) -> list[dict]:
        return _jsonl(self.out / "worklist.jsonl")

    def key(self) -> list[dict]:
        return _jsonl(self.out / "key.jsonl")

    def drawn_blocks(self) -> dict[str, set[str]]:
        """The block of every drawn item, by stratum."""
        origins = {row["id"]: _origin(row)[0] for row in self.worklist()}
        blocks: dict[str, set[str]] = {}
        for row in self.key():
            n = origins[row["id"]]
            block = next(name for name, span in self.blocks.items() if n in span)
            blocks.setdefault(row["stratum"], set()).add(block)
        return blocks


def _read(path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return [json.loads(line) for line in stream]


def _not_graded(row: dict) -> dict:
    return {**row, "graded": False, "flagged": False, "outcome": "not_graded", "causes": [],
            "consonants": []}


# --- the spec ------------------------------------------------------------------------------------


def test_round_3_declares_the_preregistrations_disclosed_counts_and_round_2_none():
    assert asp.CONSONANT3.catch_loss_disclosure == {"c95": 3505, "reference": 2623, "both": 1624}
    assert asp.CONSONANT2.catch_loss_disclosure == {}


# --- the strata ----------------------------------------------------------------------------------


def test_the_populations_match_the_hand_counts(tmp_path):
    look = Round3Audit(tmp_path)

    summary = look.draw()

    assert json.loads((look.out / "summary.json").read_text(encoding="utf-8")) == summary
    assert summary["choice_commit"] == _head(look.repo)
    assert summary["sampled"] == {"removed": 100, "added": 50}
    # block A: C95's flag by n % 3; block D: the round-3 flag by n % 2
    assert summary["populations"]["removed"] == {
        "words": 130, "causes": {"tashkeel": 44, "other_consonant": 86}}
    assert summary["populations"]["added"] == {
        "words": 60, "causes": {"tashkeel": 30, "other_consonant": 30}}
    strata = [row["stratum"] for row in look.key()]
    assert (strata.count("removed"), strata.count("added")) == (100, 50)
    for row in look.key():
        assert row["probability"] == pytest.approx(100 / 130 if row["stratum"] == "removed"
                                                   else 50 / 60)


def test_a_word_c95_flags_and_r2_does_not_is_in_neither_stratum(tmp_path):
    """Whole strata are drawn, so every block's membership shows: A is removed, D is added,
    and B (C95 only), C, E, F and G are in neither."""
    look = Round3Audit(tmp_path, removed=30, added=20)

    summary = look.draw()

    assert summary["populations"]["removed"]["words"] == 30
    assert summary["populations"]["added"]["words"] == 20
    assert summary["sampled"] == {"removed": 30, "added": 20}
    assert look.drawn_blocks() == {"removed": {"A"}, "added": {"D"}}
    assert {_origin(row)[1] for row in look.worklist()} == {0}
    assert len(look.worklist()) == 50


def test_a_word_the_r2_model_did_not_grade_is_in_neither_stratum(tmp_path):
    def ungraded(rows):
        return [_not_graded(row) if row["model"] == REFERENCE and row["position"] == 0
                and row["key"] in (_item(0).key, _item(195).key) else row for row in rows]

    look = Round3Audit(tmp_path, edit_reference=ungraded)

    summary = look.draw()

    assert summary["populations"]["removed"]["words"] == 129  # item 0 is in A
    assert summary["populations"]["added"]["words"] == 59  # item 195 is in D


def test_a_word_the_round_3_model_did_not_grade_is_in_neither_stratum(tmp_path):
    def ungraded(rows):
        return [_not_graded(row) if row["model"] != asp.C95 and row["position"] == 0
                and row["key"] in (_item(0).key, _item(195).key) else row for row in rows]

    look = Round3Audit(tmp_path, edit=ungraded)

    summary = look.draw()

    assert summary["populations"]["removed"]["words"] == 129
    assert summary["populations"]["added"]["words"] == 59


def test_a_stratum_smaller_than_its_sample_is_taken_whole(tmp_path):
    look = Round3Audit(tmp_path, removed=30, added=20)

    summary = look.draw()

    assert summary["sampled"] == {"removed": 30, "added": 20}
    assert {row["probability"] for row in look.key()} == {1.0}


def test_an_empty_added_stratum_is_drawn_as_empty(tmp_path):
    look = Round3Audit(tmp_path, removed=30, added=0)

    summary = look.draw()

    assert summary["sampled"] == {"removed": 30, "added": 0}
    assert {row["stratum"] for row in look.key()} == {"removed"}


def test_draw_refuses_an_empty_removed_stratum(tmp_path):
    look = Round3Audit(tmp_path, removed=0, added=10)

    with pytest.raises(REFUSED, match="removed"):
        look.draw()
    assert not look.out.exists()


# --- the record ----------------------------------------------------------------------------------


def test_the_key_records_all_three_models_grades(tmp_path):
    look = Round3Audit(tmp_path)
    look.draw()
    origins = {row["id"]: _origin(row)[0] for row in look.worklist()}

    for row in look.key():
        assert set(row) == {"id", "stratum", "probability", "c95", "reference", "candidate"}
        n = origins[row["id"]]
        assert row["reference"] == ({"outcome": "flagged", "causes": ["other_consonant"],
                                     "consonants": [["other_consonant", "ق→ك"]]}
                                    if row["stratum"] == "removed" else
                                    {"outcome": "not_flagged", "causes": [], "consonants": []})
        if row["stratum"] == "removed":
            assert row["c95"]["outcome"] == "flagged"
            assert row["c95"]["causes"] == list(_c95_flag(n).causes)
            assert row["candidate"]["outcome"] == "not_flagged"
        else:
            assert row["c95"]["outcome"] == "not_flagged"
            assert row["candidate"]["outcome"] == "flagged"
            assert row["candidate"]["causes"] == list(_r3_flag(n).causes)


def test_the_seed_is_round_3s_and_the_ids_derive_from_it(tmp_path):
    look = Round3Audit(tmp_path)

    summary = look.draw()

    assert summary["seed"] == SEED
    assert SEED != audit.SEED
    for row in look.worklist():
        expected = hashlib.sha256(
            f"{SEED}:{row['split']}:{row['shard_file']}:{row['row']}:{row['word_index']}"
            .encode("utf-8")).hexdigest()[:16]
        assert row["id"] == expected


def test_draw_is_deterministic_from_the_seed_and_the_inputs(tmp_path):
    first, second = Round3Audit(tmp_path / "a"), Round3Audit(tmp_path / "b")

    one, two = first.draw(), second.draw()

    for name in ("worklist.jsonl", "key.jsonl"):
        assert (first.out / name).read_bytes() == (second.out / name).read_bytes()
    assert {**one, "choice_commit": None} == {**two, "choice_commit": None}


def test_a_worklist_row_holds_only_what_the_page_may_know(tmp_path):
    look = Round3Audit(tmp_path)
    look.draw()

    rows = look.worklist()

    assert [row["id"] for row in rows] == sorted(row["id"] for row in rows)
    assert len({row["id"] for row in rows}) == 150
    required = {"id", "split", "shard_file", "row", "audio_filename", "audio_sha256",
                "start_sample", "end_sample", "surah_ayah", "word_index", "word"}
    secret = {"stratum", "probability", "c95", "reference", "candidate", "causes", "consonants",
              "outcome", "flagged", "graded", "model"}
    for row in rows:
        assert set(row) == required
        assert not secret & set(row)
        assert row["word"] == WORDS[row["word_index"]]
    assert not {"removed", "added"} & set(json.dumps(rows))


def test_round_2s_draw_is_unchanged_in_shape(tmp_path):
    look = Round2Audit(tmp_path)

    summary = look.draw()

    assert set(summary) == {"seed", "choice_commit", "populations", "sampled"}
    assert summary["seed"] == audit.SEED == "round2-catch-loss-2026-10"
    for row in look.key():
        assert set(row) == {"id", "stratum", "probability", "c95", "candidate"}


# --- the refusals --------------------------------------------------------------------------------


def test_draw_refuses_when_the_r2_grades_file_is_not_committed(tmp_path):
    look = Round3Audit(tmp_path, uncommitted=(REFERENCE_EXPORT,))

    with pytest.raises(REFUSED, match="look_grades_r2"):
        look.draw()
    assert not look.out.exists()


def test_draw_refuses_when_the_r2_grades_file_holds_no_r2_grades(tmp_path):
    look = Round3Audit(tmp_path, edit_reference=lambda rows: [
        row for row in rows if row["model"] != REFERENCE])

    with pytest.raises(REFUSED, match="r2"):
        look.draw()
    assert not look.out.exists()


def test_draw_refuses_a_c95_grade_the_r2_file_does_not_repeat_word_for_word(tmp_path):
    def regraded(rows):
        changed = False
        for row in rows:
            if row["model"] == asp.C95 and row["key"] == _item(130).key and row["position"] == 0:
                row.update(flagged=False, outcome="not_flagged", causes=[], consonants=[])
                changed = True
        assert changed
        return rows

    look = Round3Audit(tmp_path, edit_reference=regraded)

    with pytest.raises(REFUSED, match="C95|c95"):
        look.draw()
    assert not look.out.exists()


def test_draw_refuses_a_missing_c95_word_in_the_r2_file(tmp_path):
    look = Round3Audit(tmp_path, edit_reference=lambda rows: [
        row for row in rows
        if not (row["model"] == asp.C95 and row["key"] == _item(3).key and row["position"] == 0)])

    with pytest.raises(REFUSED):
        look.draw()
    assert not look.out.exists()


@pytest.mark.parametrize("name", ["c95", "reference", "both"])
@pytest.mark.parametrize("off", [-1, 1])
def test_draw_refuses_when_a_disclosed_count_is_not_reproduced(tmp_path, name, off):
    probe = Round3Audit(tmp_path / "probe")
    wrong = {**probe.disclosed, name: probe.disclosed[name] + off}
    look = Round3Audit(tmp_path / "look", disclosure=wrong)

    with pytest.raises(REFUSED, match="disclosed|3,505|count"):
        look.draw()
    assert not look.out.exists()


def test_draw_passes_when_the_disclosed_counts_are_reproduced(tmp_path):
    probe = Round3Audit(tmp_path / "probe")
    look = Round3Audit(tmp_path / "look", disclosure=probe.disclosed)

    summary = look.draw()

    assert summary["sampled"] == {"removed": 100, "added": 50}


def test_draw_refuses_while_the_round_3_pre_registration_pin_is_unfilled(tmp_path):
    look = Round3Audit(tmp_path)
    unfilled = dataclasses.replace(
        look.spec, pins=dataclasses.replace(look.spec.pins, prereg_sha256=None))

    with pytest.raises(REFUSED, match="pre-registration"):
        look.draw(unfilled)
    assert not look.out.exists()


def test_draw_refuses_to_redraw_a_round_3_worklist(tmp_path):
    look = Round3Audit(tmp_path)
    look.draw()
    written = {path.name: path.read_bytes() for path in look.out.iterdir()}

    with pytest.raises(REFUSED, match="exists"):
        look.draw()
    assert {path.name: path.read_bytes() for path in look.out.iterdir()} == written


# --- the CLI -------------------------------------------------------------------------------------


@pytest.fixture
def cli(monkeypatch):
    """``main()`` with ``draw`` and ``score`` recorded instead of run."""
    calls = []

    def draw(spec, *, uthmani_words, tranche):
        calls.append(("draw", spec, tranche))
        return {}

    def score(directory, *, spec=asp.CONSONANT2):
        calls.append(("score", spec, directory))
        return {"verdict": "pass", "removed": {"real": 0, "decisive": 0}}

    monkeypatch.setattr(audit, "draw", draw)
    monkeypatch.setattr(audit, "score", score)

    def run(*argv):
        calls.clear()
        monkeypatch.setattr(sys, "argv", ["catch_loss_audit", *argv])
        audit.main()
        return calls[0]

    return run


def test_draw_takes_the_round_from_spec(cli):
    command, spec, tranche = cli("draw", "--spec", "consonant3", "--tranche", "look.jsonl")

    assert (command, spec.name, spec.root, tranche) == (
        "draw", "consonant3", asp.CONSONANT3.root, Path("look.jsonl"))
    assert spec.models == asp.CONSONANT3.models


def test_draw_defaults_to_round_2_and_keeps_root_an_override(cli, tmp_path):
    assert cli("draw", "--tranche", "look.jsonl")[1].root == asp.CONSONANT2.root
    spec = cli("draw", "--spec", "consonant3", "--root", str(tmp_path), "--tranche", "t")[1]
    assert (spec.name, spec.root) == ("consonant3", tmp_path)


def test_score_takes_the_round_and_its_directory_from_spec(cli):
    command, spec, directory = cli("score", "--spec", "consonant3")
    assert (command, spec.name, directory) == (
        "score", "consonant3", asp.CONSONANT3.root / "catch_loss")

    _, spec, directory = cli("score")
    assert (spec.name, directory) == ("consonant2", asp.CONSONANT2.root / "catch_loss")


def test_an_unknown_spec_is_refused_and_a_known_one_is_not(cli):
    assert cli("draw", "--spec", "consonant3", "--tranche", "look.jsonl")[1].name == "consonant3"
    with pytest.raises(SystemExit):
        cli("draw", "--spec", "consonant9", "--tranche", "look.jsonl")


# --- the committed real files --------------------------------------------------------------------

ROOT = asp.CONSONANT3.root
#: The committed look export is never rewritten (decision, 2026-11-03).
EXPORT_SHA256 = "d97e4796af71edec45d0a19555e1367757398c0b27f73e52a02b33fecb5e5f40"
CANDIDATE = "r3_candidate_ema"


def _flagged(rows, model) -> set[tuple[str, int]]:
    return {(row["key"], row["position"]) for row in rows if row["model"] == model
            and row["flagged"]}


def _graded(rows, model) -> set[tuple[str, int]]:
    return {(row["key"], row["position"]) for row in rows if row["model"] == model
            and row["graded"]}


def test_the_committed_look_export_is_untouched_and_carries_no_r2():
    assert hashlib.sha256((ROOT / EXPORT).read_bytes()).hexdigest() == EXPORT_SHA256
    assert {row["model"] for row in read_jsonl_gz(ROOT / EXPORT)} == {asp.C95, CANDIDATE}


def test_the_committed_r2_grades_reproduce_the_disclosed_counts():
    rows = read_jsonl_gz(ROOT / REFERENCE_EXPORT)

    assert {row["model"] for row in rows} == {asp.C95, REFERENCE}
    c95, r2 = _flagged(rows, asp.C95), _flagged(rows, REFERENCE)
    assert (len(c95), len(r2), len(c95 & r2)) == (3505, 2623, 1624)


def test_the_committed_r2_file_repeats_the_exports_c95_grades_word_for_word():
    export = [row for row in read_jsonl_gz(ROOT / EXPORT) if row["model"] == asp.C95]
    repeated = [row for row in read_jsonl_gz(ROOT / REFERENCE_EXPORT) if row["model"] == asp.C95]

    assert repeated == export
    assert {(row["key"], row["position"]) for row in repeated} == {
        (row["key"], row["position"]) for row in read_jsonl_gz(ROOT / REFERENCE_EXPORT)
        if row["model"] == REFERENCE}


def test_the_committed_drawn_set_matches_the_files_it_was_drawn_from():
    out = ROOT / "catch_loss"
    summary = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    worklist, key = _jsonl(out / "worklist.jsonl"), _jsonl(out / "key.jsonl")
    export = read_jsonl_gz(ROOT / EXPORT)
    reference_rows = read_jsonl_gz(ROOT / REFERENCE_EXPORT)
    # the population is the words graded by all three models (prereg: "graded by C95, the
    # round-2 EMA and the chosen round-3 weights")
    graded = (_graded(export, asp.C95) & _graded(export, CANDIDATE)
              & _graded(reference_rows, REFERENCE))
    c95, r3 = _flagged(export, asp.C95) & graded, _flagged(export, CANDIDATE) & graded
    r2 = _flagged(reference_rows, REFERENCE) & graded
    removed, added = len(c95 & r2 - r3), len(r3 - c95 - r2)

    assert summary["seed"] == SEED
    assert summary["populations"]["removed"]["words"] == removed
    assert summary["populations"]["added"]["words"] == added
    assert summary["sampled"] == {"removed": min(100, removed), "added": min(50, added)}
    assert [row["id"] for row in worklist] == sorted(row["id"] for row in worklist)
    assert {row["id"] for row in worklist} == {row["id"] for row in key}
    assert len(key) == sum(summary["sampled"].values())
    assert {row["stratum"] for row in key} <= {"removed", "added"}
    for row in key:
        assert set(row) == {"id", "stratum", "probability", "c95", "reference", "candidate"}
        flags = [grade["outcome"] != "not_flagged" for grade in
                 (row["c95"], row["reference"], row["candidate"])]
        assert flags == ([True, True, False] if row["stratum"] == "removed"
                         else [False, False, True])


@pytest.mark.parametrize("name", ["worklist.jsonl", "key.jsonl"])
def test_the_round_3_drawn_files_are_not_gitignored(name):
    """AGENTS.md: ``*.jsonl`` is ignored by default, so each label file needs a ``!`` entry."""
    ignored = subprocess.run(
        ["git", "check-ignore", "-q", str(ROOT / "catch_loss" / name)],
        cwd=ROOT, capture_output=True).returncode

    assert ignored == 1

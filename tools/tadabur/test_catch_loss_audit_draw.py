"""The round-2 catch-loss audit's worklist builder (#225): ``tadabur.catch_loss_audit.draw``.

``draw(spec, uthmani_words=..., tranche=...)`` reads the look sitting's committed per-word
export (``<spec.root>/look_grades.jsonl.gz``: C95's and the chosen candidate's kept grades on
tlog look gating), the committed live/EMA choice and the look tranche manifest, and writes, in
``<spec.root>/catch_loss/``:

* ``worklist.jsonl`` — what the page may know: per sampled word an opaque ``id``, ``split``,
  ``shard_file``, ``row``, ``audio_filename``, ``audio_sha256``, the clip's ``start_sample`` and
  ``end_sample`` (recorded; the page plays the whole clip), ``surah_ayah``, the 0-based Uthmani
  ``word_index`` and the word's ``word`` text, sorted by ``id``;
* ``key.jsonl`` — never served: per item ``id``, ``stratum`` (``removed`` or ``added``),
  ``probability`` and each model's grade as ``c95`` / ``candidate``, each ``{outcome, causes,
  consonants}``, sorted by ``id``;
* ``summary.json`` — ``seed``, ``choice_commit`` (the commit holding ``choice.json``),
  ``populations[stratum] = {"words", "causes"}`` and ``sampled[stratum]``. A stratum's causes
  are those of the model that flags its words (C95's for ``removed``, the candidate's for
  ``added``), a word counted once per cause.

``draw`` returns the summary. It refuses (``SystemExit`` or ``ValueError``) unless the
pre-registration is committed and pinned, ``choice.json`` and the export are committed, every
export row is a look gating row of the tranche manifest, and no worklist exists yet.

The sitting is real: it is made by :func:`training.aug_readout.run_look` on synthetic grades in a
throwaway git repository, so the export and the committed choice are the ones the audit reads.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import subprocess

import pytest

from tadabur import catch_loss_audit as audit
from training import aug_readout as ar
from training import aug_readout_gate as ag
from training import round2_fixtures as fx

REFUSED = (SystemExit, ValueError)
BOTH = 20  # items whose second word both models flag: in neither population
WORDS = ("بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ", "ٱلْحَمْدُ", "لِلَّهِ", "رَبِّ",
         "ٱلْعَٰلَمِينَ", "مَٰلِكِ", "يَوْمِ")
FIRST_WORD_INDEX = 7  # round2_fixtures.word_grade: position p is canonical word 7 + p
EXPORT = ar.LOOK_GRADES


def _words(surah_ayah: str) -> list[str]:
    return list(WORDS)


def _item(n: int):
    item = fx.tlog_item(n, 1 + n % 5)
    return dataclasses.replace(item, audio_sha256=hashlib.sha256(f"clip{n}".encode()).hexdigest())


class _Sources(fx.TlogSources):
    """The look set with each item on its own shard row: item ``n`` is row ``n % 10`` of shard
    ``clean-<n // 10>``."""

    def tlog(self, role, models):
        data = super().tlog(role, models)
        for lines in data.decodes.values():
            for n, item in enumerate(self.items):
                lines[item.key].update(audio_sha256=item.audio_sha256,
                                       shard_file=f"clean-{n // 10:05d}", row=n % 10)
        return data


def _origin(row: dict) -> tuple[int, int]:
    """The item index and word position a worklist row came from."""
    item = int(row["shard_file"].rsplit("-", 1)[1]) * 10 + row["row"]
    return item, row["word_index"] - FIRST_WORD_INDEX


def _jsonl(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _head(repo) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()


class LookAudit:
    """A committed look sitting in which the first ``removed`` items' first word is flagged by
    C95 only (a tashkeel, س→ص or س→ش flag by ``n % 3``) and the next ``added`` items' first word
    by the candidate only (tashkeel, or ث→س, by ``n % 2``). The first :data:`BOTH` items' second
    word is flagged by both. ``edit`` rewrites the export's rows before the commit;
    ``uncommitted`` names the look files left out of it."""

    def __init__(self, root, *, removed=130, added=60, edit=None, uncommitted=()):
        self.repo = fx.init_repo(root / "repo")
        self.prereg = fx.commit_prereg(self.repo)
        self.spec = fx.filled_spec(root=self.repo / "readout", prereg=self.prereg)
        self.layout = ag.Layout(self.spec.root)
        self.layout.root.mkdir()
        self.layout.models.write_text(json.dumps(fx.look_manifest(self.spec)), encoding="utf-8")
        items = [_item(n) for n in range(removed + added)]

        def flags(model, n, position):
            if position == 1:
                both = model == "c95" or model.startswith("r2_candidate")
                return fx.consonant_flag("ق→ك") if both and n < BOTH else None
            if model == "c95":
                return (fx.TASHKEEL_FLAG, fx.consonant_flag("س→ص"),
                        fx.consonant_flag("س→ش"))[n % 3] if n < removed else None
            if model.startswith("r2_candidate") and removed <= n:
                return fx.TASHKEEL_FLAG if n % 2 == 0 else fx.consonant_flag("ث→س")
            return None

        self.items = items
        ar.run_look(_Sources(root / "files", items, flags, spec=self.spec), self.layout,
                    spec=self.spec)
        self.export = self.layout.root / EXPORT
        if edit is not None:
            with gzip.open(self.export, "rt", encoding="utf-8") as stream:
                rows = [json.loads(line) for line in stream]
            ar.write_look_grades(self.export, edit(rows))
        names = [self.layout.look.name, self.layout.choice.name, self.layout.claim("look").name,
                 self.export.name]
        fx.git(self.layout.root, "add", *(n for n in names if n not in uncommitted))
        fx.git(self.layout.root, "commit", "-qm", "the look")
        self.tranche = root / "look.jsonl"
        self.tranche.write_text("".join(json.dumps({
            "audio_path": item.audio_filename, "audio_sha256": item.audio_sha256,
            "split": "clean", "shard_file": f"clean-{n // 10:05d}", "row": n % 10,
            "stratum": "gating"}) + "\n" for n, item in enumerate(items)), encoding="utf-8")
        self.out = self.spec.root / "catch_loss"

    def draw(self, spec=None) -> dict:
        return audit.draw(spec or self.spec, uthmani_words=_words, tranche=self.tranche)

    def worklist(self) -> list[dict]:
        return _jsonl(self.out / "worklist.jsonl")

    def key(self) -> list[dict]:
        return _jsonl(self.out / "key.jsonl")


# --- the sample and the record of it -------------------------------------------------------------


def test_the_seed_is_the_pre_registered_one():
    assert audit.SEED == "round2-catch-loss-2026-10"


def test_draw_takes_100_removed_and_50_added_and_records_the_populations(tmp_path):
    look = LookAudit(tmp_path)

    summary = look.draw()

    assert json.loads((look.out / "summary.json").read_text(encoding="utf-8")) == summary
    assert summary["seed"] == audit.SEED
    assert summary["choice_commit"] == _head(look.repo)
    assert summary["sampled"] == {"removed": 100, "added": 50}
    assert summary["populations"]["removed"] == {
        "words": 130, "causes": {"tashkeel": 44, "other_consonant": 86}}
    assert summary["populations"]["added"] == {
        "words": 60, "causes": {"tashkeel": 30, "other_consonant": 30}}
    strata = [row["stratum"] for row in look.key()]
    assert (strata.count("removed"), strata.count("added")) == (100, 50)


def test_each_item_records_its_sampling_probability(tmp_path):
    look = LookAudit(tmp_path)
    look.draw()
    for row in look.key():
        assert row["probability"] == pytest.approx(100 / 130 if row["stratum"] == "removed"
                                                   else 50 / 60)


def test_removed_is_what_c95_flags_and_the_candidate_does_not_and_added_the_reverse(tmp_path):
    look = LookAudit(tmp_path)
    look.draw()
    origins = {row["id"]: _origin(row) for row in look.worklist()}
    for row in look.key():
        n, position = origins[row["id"]]
        assert position == 0  # the words both models flag, and the unflagged ones, are out
        if row["stratum"] == "removed":
            assert n < 130
            assert row["c95"] == {
                "outcome": "flagged",
                "causes": [("tashkeel", "other_consonant", "other_consonant")[n % 3]],
                "consonants": [[], [["other_consonant", "س→ص"]],
                               [["other_consonant", "س→ش"]]][n % 3]}
            assert (row["candidate"]["outcome"], row["candidate"]["causes"]) == ("not_flagged", [])
        else:
            assert 130 <= n < 190
            assert row["candidate"]["outcome"] == "flagged"
            assert row["candidate"]["causes"] == [("tashkeel", "other_consonant")[n % 2]]
            assert row["c95"]["outcome"] == "not_flagged"


def test_a_word_graded_by_only_one_model_is_in_neither_population(tmp_path):
    def ungraded(rows):
        for row in rows:
            first = row["position"] == 0
            if first and row["key"] == _item(0).key and row["model"] != "c95":
                row.update(graded=False, flagged=False, outcome="not_graded",
                           causes=[], consonants=[])
            if first and row["key"] == _item(130).key and row["model"] == "c95":
                row.update(graded=False, outcome="not_graded")
        return rows

    look = LookAudit(tmp_path, edit=ungraded)

    summary = look.draw()

    assert summary["populations"]["removed"] == {
        "words": 129, "causes": {"tashkeel": 43, "other_consonant": 86}}
    assert summary["populations"]["added"] == {
        "words": 59, "causes": {"tashkeel": 29, "other_consonant": 30}}
    assert sorted({row["probability"] for row in look.key()}) == pytest.approx([100 / 129, 50 / 59])


def test_a_stratum_smaller_than_its_sample_is_taken_whole(tmp_path):
    look = LookAudit(tmp_path, removed=30, added=20)

    summary = look.draw()

    assert summary["sampled"] == {"removed": 30, "added": 20}
    assert {row["probability"] for row in look.key()} == {1.0}
    assert len(look.worklist()) == 50


def test_a_flag_in_the_real_outcome_vocabulary_is_still_a_flag(tmp_path):
    def real_outcomes(rows):
        for row in rows:
            if row["flagged"]:  # a word grade's outcome is never "flagged" in the real export
                row["outcome"] = ("flagged_unattributed", "flagged_elsewhere")[row["row"] % 2]
        return rows

    look = LookAudit(tmp_path, edit=real_outcomes)

    summary = look.draw()

    assert summary["populations"]["removed"] == {
        "words": 130, "causes": {"tashkeel": 44, "other_consonant": 86}}
    assert summary["populations"]["added"] == {
        "words": 60, "causes": {"tashkeel": 30, "other_consonant": 30}}
    assert summary["sampled"] == {"removed": 100, "added": 50}


def test_an_empty_added_stratum_is_drawn_as_empty(tmp_path):
    look = LookAudit(tmp_path, removed=30, added=0)

    summary = look.draw()

    assert summary["sampled"] == {"removed": 30, "added": 0}
    assert {row["stratum"] for row in look.key()} == {"removed"}


def test_draw_is_deterministic_from_the_seed_and_the_inputs(tmp_path):
    first, second = LookAudit(tmp_path / "a"), LookAudit(tmp_path / "b")

    one, two = first.draw(), second.draw()

    for name in ("worklist.jsonl", "key.jsonl"):
        assert (first.out / name).read_bytes() == (second.out / name).read_bytes()
    assert {**one, "choice_commit": None} == {**two, "choice_commit": None}


# --- the worklist is what the page may know ------------------------------------------------------


def test_a_worklist_row_holds_the_provenance_and_the_word_and_nothing_about_the_models(tmp_path):
    look = LookAudit(tmp_path)
    look.draw()

    rows = look.worklist()

    assert [row["id"] for row in rows] == sorted(row["id"] for row in rows)
    assert len({row["id"] for row in rows}) == 150
    required = {"id", "split", "shard_file", "row", "audio_filename", "audio_sha256",
                "start_sample", "end_sample", "surah_ayah", "word_index", "word"}
    secret = {"stratum", "probability", "weight", "c95", "candidate", "causes", "consonants",
              "outcome", "flagged", "graded", "model"}
    for row in rows:
        assert required <= set(row)
        assert not secret & set(row)
        n, position = _origin(row)
        item = look.items[n]
        assert (row["split"], row["audio_filename"], row["audio_sha256"]) == (
            "clean", item.audio_filename, item.audio_sha256)
        assert (row["start_sample"], row["end_sample"]) == (item.start_sample, item.end_sample)
        assert row["surah_ayah"] == f"{item.surah}:{item.ayah}"
        assert row["word"] == WORDS[row["word_index"]]
    assert not {"removed", "added"} & set(json.dumps(rows))


# --- the refusals --------------------------------------------------------------------------------


def test_draw_refuses_while_the_pre_registration_pin_is_unfilled(tmp_path):
    look = LookAudit(tmp_path)
    unfilled = dataclasses.replace(
        look.spec, pins=dataclasses.replace(look.spec.pins, prereg_sha256=None))

    with pytest.raises(REFUSED, match="pre-registration"):
        look.draw(unfilled)
    assert not look.out.exists()


def test_draw_refuses_a_pre_registration_edited_since_its_commit(tmp_path):
    look = LookAudit(tmp_path)
    look.prereg[0].write_text("edited after the freeze\n", encoding="utf-8")

    with pytest.raises(REFUSED, match="pre-registration"):
        look.draw()
    assert not look.out.exists()


def test_draw_refuses_without_a_committed_choice(tmp_path):
    look = LookAudit(tmp_path, uncommitted=("choice.json",))

    with pytest.raises(REFUSED, match="choice"):
        look.draw()
    assert not look.out.exists()


def test_draw_refuses_an_export_that_is_not_committed(tmp_path):
    look = LookAudit(tmp_path, uncommitted=(EXPORT,))

    with pytest.raises(REFUSED, match="look_grades"):
        look.draw()
    assert not look.out.exists()


def test_draw_refuses_a_row_that_is_not_a_look_gating_row(tmp_path):
    look = LookAudit(tmp_path)
    rows = _jsonl(look.tranche)
    rows[3]["stratum"] = "reported:below_bar"
    look.tranche.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    with pytest.raises(REFUSED, match="gating"):
        look.draw()
    assert not look.out.exists()


def test_draw_refuses_an_empty_removed_stratum(tmp_path):
    look = LookAudit(tmp_path, removed=0, added=10)

    with pytest.raises(REFUSED, match="removed"):
        look.draw()
    assert not look.out.exists()


def test_draw_refuses_to_redraw_over_a_worklist_and_leaves_it_untouched(tmp_path):
    look = LookAudit(tmp_path)
    look.draw()
    written = {path.name: path.read_bytes() for path in look.out.iterdir()}

    with pytest.raises(REFUSED, match="exists"):
        look.draw()
    assert {path.name: path.read_bytes() for path in look.out.iterdir()} == written


def test_a_worklist_alone_is_enough_to_refuse_a_redraw(tmp_path):
    look = LookAudit(tmp_path)
    look.draw()
    (look.out / "key.jsonl").unlink()
    (look.out / "summary.json").unlink()

    with pytest.raises(REFUSED, match="exists"):
        look.draw()
    assert not (look.out / "key.jsonl").exists()

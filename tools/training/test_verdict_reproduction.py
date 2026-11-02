"""The pooled verdict runs #95's reproduction check before it reads a confirm decode (#249).

The pre-registration's reproduction check (not a bar) re-grades C95 on #95's Tarteel v1 confirm
(1,971 / 41,572) and qdat (670 / 11,502) and stops on any mismatch. Round 1 runs it inside its
claim (``aug_readout.run_confirm``); the pooled verdict must too, and must record the SHA-256 of
every file the check opened. The check itself is the real :func:`training.aug_readout.reproduction`
in the slow tier; the fast tier plants its totals and its input files around the verdict, whose
fixtures are ``test_consonant2_verdict.py``'s.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from training import aug_readout as ar
from training import aug_readout_spec as asp
from training.test_consonant2_verdict import ConfirmSources, MissingInput, _opened, _verdict

#: #95's four-model Tarteel-confirm and qdat decodes, and the database its harness grades with.
DECODE_NAMES = tuple(f"{name}.{model}.jsonl.gz" for name in ("tarteel_v1.confirm", "qdat")
                     for model in ar.REPRODUCTION_MODELS)
INPUT_NAMES = (*DECODE_NAMES, "quran.db")


@pytest.fixture
def cv():
    from training import consonant2_verdict

    return consonant2_verdict


class SpiedSources(ConfirmSources):
    """The confirm sources, every read (tranche, Tarteel confirm half, Tadabur dev) logged to
    ``events``, which the reproduction check's own calls share."""

    def __init__(self, spec, events: list[str]):
        super().__init__(spec)
        self.events = events

    def tlog_views(self, role, models):
        self.events.append(f"read {role}")
        return super().tlog_views(role, models)

    def tarteel(self, which_half, models):
        self.events.append(f"read tarteel {which_half}")
        return super().tarteel(which_half, models)

    def tadabur(self, models):
        self.events.append("read tadabur")
        return super().tadabur(models)


class PlantedCheck:
    """What the monkeypatched check reports: the totals it returns, the files it says it opened,
    and the calls it saw in order."""

    def __init__(self, tmp_path: Path, monkeypatch, counts=None):
        self.events: list[str] = []
        self.sources: list = []
        self.counts = ar.REPRODUCTION if counts is None else counts
        directory = tmp_path / "reproduction_inputs"
        directory.mkdir()
        self.files = []
        for name in INPUT_NAMES:
            path = directory / name
            path.write_bytes(f"the bytes of {name}".encode())
            self.files.append(path)
        monkeypatch.setattr(ar, "reproduction", self._reproduction)
        monkeypatch.setattr(ar, "reproduction_inputs", lambda *args, **kwargs: list(self.files),
                            raising=False)

    def _reproduction(self, sources):
        self.events.append("reproduction")
        self.sources.append(sources)
        return self.counts

    @property
    def digests(self) -> dict[str, str]:
        return {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in self.files}


def _perturbed(set_name: str) -> dict:
    counts = json.loads(json.dumps(ar.REPRODUCTION))
    counts[set_name]["flagged"] += 1
    return counts


# --- a mismatch stops the sitting before any confirm decode is read -----------------------------


@pytest.mark.parametrize("set_name", sorted(ar.REPRODUCTION))
def test_a_mismatch_exits_before_any_confirm_decode_is_read(cv, tmp_path, monkeypatch, set_name):
    check = PlantedCheck(tmp_path, monkeypatch, _perturbed(set_name))
    round_ = _opened(tmp_path, monkeypatch, cv)
    sources = SpiedSources(round_.spec, check.events)

    def opened(*args, **kwargs):
        raise AssertionError("a decode file was opened after a failed reproduction check")

    from training import external_corpus

    monkeypatch.setattr(external_corpus, "read_decode_file", opened)
    monkeypatch.setattr(ar, "read_readout", opened)

    with pytest.raises(SystemExit, match="reproduction check failed"):
        _verdict(cv, round_, sources)

    assert check.events == ["reproduction"]
    assert sources.read == []


def test_a_mismatch_leaves_the_claim_and_writes_no_report_verdict_or_doc(cv, tmp_path,
                                                                         monkeypatch):
    PlantedCheck(tmp_path, monkeypatch, _perturbed("qdat"))
    round_ = _opened(tmp_path, monkeypatch, cv)

    with pytest.raises(SystemExit, match="reproduction check failed"):
        _verdict(cv, round_)

    assert round_.layout.claim("confirm").exists()  # the read was opened
    assert not round_.layout.report.exists()
    assert not round_.layout.verdict.exists()
    assert not round_.spec.doc_path.exists()


# --- a pass is recorded with its inputs' digests ------------------------------------------------


def test_the_check_runs_first_inside_the_sitting_on_95_s_own_sources(cv, tmp_path, monkeypatch):
    check = PlantedCheck(tmp_path, monkeypatch)
    round_ = _opened(tmp_path, monkeypatch, cv)
    _verdict(cv, round_, SpiedSources(round_.spec, check.events))

    assert check.events[0] == "reproduction"
    assert check.events.count("reproduction") == 1
    assert {event for event in check.events[1:]} >= {"read confirm.1", "read confirm.2"}
    (sources,) = check.sources
    assert isinstance(sources, ar.Sources) and sources.spec is asp.AUG95  # never this round's


def test_the_report_records_the_passed_check_with_one_sha_256_per_opened_file(cv, tmp_path,
                                                                             monkeypatch):
    check = PlantedCheck(tmp_path, monkeypatch)
    round_ = _opened(tmp_path, monkeypatch, cv)
    sources = ConfirmSources(round_.spec)
    returned = _verdict(cv, round_, sources)

    written = json.loads(round_.layout.report.read_text(encoding="utf-8"))
    for report in (returned, written):
        record = report["reproduction"]
        assert record["passed"] is True
        assert record["counts"] == ar.REPRODUCTION
        assert record["expected"] == ar.REPRODUCTION
        assert record["order"] == "before the confirm read"
        assert record["harness"] == sources.harness_record
        assert record["inputs"] == check.digests
        assert len(record["inputs"]) == len(INPUT_NAMES)
        assert all(len(digest) == 64 for digest in record["inputs"].values())


def test_a_sitting_with_a_declared_input_absent_still_records_the_check(cv, tmp_path,
                                                                       monkeypatch):
    check = PlantedCheck(tmp_path, monkeypatch)
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_, MissingInput(round_.spec))

    assert "unreadable" in report
    assert report["reproduction"]["passed"] is True
    assert report["reproduction"]["inputs"] == check.digests


# --- the real inputs ----------------------------------------------------------------------------


def _harness():
    from training.muraja_policy import Harness

    return Harness.locate()


def test_the_inputs_are_the_files_the_check_reads_and_the_database(monkeypatch):
    """Collected from what the readers open, not from a hand list: #95's eight decode files, in
    its readout directory and the committed baselines, and the harness's ``quran.db``."""
    from tadabur.tarteel_split import CONFIRM
    from training import external_corpus

    harness = _harness()
    if harness is None:
        pytest.skip("the Muraja harness is not built here")
    opened: list[Path] = []
    read = external_corpus.read_decode_file

    def spy(path, *args, **kwargs):
        opened.append(Path(path))
        return read(path, *args, **kwargs)

    monkeypatch.setattr(external_corpus, "read_decode_file", spy)
    sources = ar.Sources()
    sources.tarteel(CONFIRM, ar.REPRODUCTION_MODELS)
    sources.qdat(ar.REPRODUCTION_MODELS)

    inputs = ar.reproduction_inputs()
    assert len(opened) == len(DECODE_NAMES) == 8
    assert sorted(inputs, key=str) == sorted([*opened, harness.quran_db], key=str)
    assert sorted(path.name for path in inputs) == sorted(INPUT_NAMES)


@pytest.mark.slow
def test_the_verdict_runs_the_real_check_and_digests_the_committed_inputs(cv, tmp_path,
                                                                          monkeypatch):
    from training.aug_readout_gate import input_name, sha256_file

    if _harness() is None:
        pytest.skip("the Muraja harness is not built here")
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_)

    record = report["reproduction"]
    assert record["passed"] is True
    assert record["counts"] == {"tarteel_v1_confirm": {"flagged": 1971, "graded": 41572},
                                "qdat": {"flagged": 670, "graded": 11502}}
    assert record["inputs"] == {input_name(path): sha256_file(path)
                                for path in ar.reproduction_inputs()}
    assert len(record["inputs"]) == len(INPUT_NAMES)


# --- round 2's render is unchanged --------------------------------------------------------------


def test_the_provenance_keys_do_not_change_the_rendered_doc(cv, tmp_path, monkeypatch):
    PlantedCheck(tmp_path, monkeypatch)
    round_ = _opened(tmp_path, monkeypatch, cv)
    report = _verdict(cv, round_)
    preclaim = json.loads(round_.preclaim.read_text(encoding="utf-8"))

    legacy = {**report, "reproduction": {key: report["reproduction"][key]
                                         for key in ("passed", "counts")}}
    assert cv.render(preclaim, report) == cv.render(preclaim, legacy)
    assert "tarteel_v1_confirm: 1971 / 41572" in cv.render(preclaim, report)


def test_round_2_s_committed_record_still_renders_its_counts(cv):
    preclaim = json.loads((asp.CONSONANT2.root / "preclaim.json").read_text(encoding="utf-8"))
    committed = json.loads((asp.CONSONANT2.root / "report.json").read_text(encoding="utf-8"))

    text = cv.render(preclaim, committed)
    for line in ("- c95 tarteel_v1_confirm: 1971 / 41572", "- c95 qdat: 670 / 11502",
                 "- c178_tarteel_v1_confirm: 1837 / 41652"):
        assert line in text
    assert text == cv.render(preclaim, committed)

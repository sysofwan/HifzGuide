"""Regressions for the round-four review of the S5 arms and scorer (#311 PR2, R4.1-R4.4).

Committed beside the locked ``test_followalong_score_*`` files, never editing them:

* R4.1 — the gate's saved traces hold one row per *graded* word, so a drawn word the engine
  reported but never graded had no App cell, and ``miss_labels`` read that as M-nograde ("the
  engine never saw the word", A2) even where its checks graded it. Such a word's cell is now a
  pending row carrying every check's own status, re-graded from the clip's own saved checks
  through the same pinned harness; a replay that does not reproduce the clip's saved grades is
  refused, and a clip with no trace row at all keeps no cell.
* R4.2 — prereg §5's R-alone row is reported beside the primary attribution, with its own model
  share, and never decides.
* R4.3 — unknown M-nograde words alone still bound rule 3's share at 0 against 1, which straddles
  50% and leaves the decision to the owner.
* R4.4 — a reference suspect explained by F1 or F1-align keeps its bucket's side; only F-ref
  carries the both-sides marker.
"""

from __future__ import annotations

import gzip
import json

import pytest

from tadabur.followalong_audit.attribution import FOLLOW_ALONG, MODEL, UNKNOWN
from tadabur.followalong_audit.attribution import reference_suspects
from tadabur.followalong_audit.s5_arms import _app_cells
from tadabur.followalong_audit.s5_score import attribute_listened, evaluate_rules

ITEM = "2_170_4222931689.flac#0"
SHA = "d57ea8f4e1245417405d4675687a84c385e7b9f8b23b797cddd1d545319826ee"
WORD, BEFORE, AFTER = "بِسمِ", "مَلِكِ", "يَومِ"
REFERENCE = {"previous": BEFORE, "word": WORD, "next": AFTER}
MISMATCH = {"group_index": 0, "kind": {"type": "mismatch", "expected": "س", "heard": "ص"}}


# --- the rows a drawn word is attributed from -----------------------------------------------------
def _key_row() -> dict:
    return {"id": "0123456789abcdef", "item": ITEM, "word": 3, "surah_ayah": "2:170",
            "stratum": "S1", "cell": "other_consonant:no", "inclusion_probability": 0.04}


def _grade(quality: str = "correct", *, score: float = 1.0) -> dict:
    """A compact arm cell: its kept grade alone."""
    return {"quality": quality, "score": score, "withheld": None, "errors": []}


def _flagged(setting_text: str) -> dict:
    """An App cell the kept grade flags; one setting check decoded ``setting_text``."""
    return {"quality": "minor", "score": 0.5, "withheld": None, "errors": [MISMATCH],
            "checks": [{"hop": setting_text, "overlap": "", "flush": False,
                        "status": {"word": 3, "quality": "minor", "score": 0.5,
                                   "errors": [MISMATCH]}}],
            "setting_checks": [0]}


def _arms(app: dict | None = None, *, oracle: str | None = "correct",
          teacher: str | None = "correct", reference: dict | None = REFERENCE) -> dict:
    def both(cell):
        return {"strict": cell, "shipped_default": cell}

    return {"item": ITEM, "word": 3, "app": both(app if app is not None else _flagged(WORD)),
            "causes": {"strict": [], "shipped_default": []},
            "oracle": both(None if oracle is None else _grade(oracle)),
            "teacher": both(None if teacher is None else _grade(teacher)),
            "student": both(None), "reference": reference}


def _suspects() -> frozenset:
    observations = [{"group_key": group, "surah": 2, "ayah": 170, "word": 3,
                     "errors": [MISMATCH]} for group in ("g1", "g2", "g3")]
    return reference_suspects(observations)


# --- R4.1: a reported word the gate gave no kept grade -------------------------------------------
class _Text:
    """The engine's phoneme text for the one ayah these clips are drawn from."""

    def phonemes(self, surah: int, ayah: int) -> str:
        return "أ ب ج"


def _status(word: int, quality: str = "correct", score: float = 1.0) -> dict:
    return {"word": word, "quality": quality, "score": score, "errors": []}


def _saved_rows() -> list[dict]:
    """One clip's saved traces: word 2 graded, word 3 reported with no kept entry.

    The two checks also graded word 3 (``minor`` then ``correct``), which the gate's traces keep
    only as part of the request's check list — never as a row of its own.
    """
    checks = [{"hop": "ها", "overlap": "", "flush": False},
              {"hop": "", "overlap": "", "flush": False, "stop": True}]
    return [{"item": ITEM, "scoring": scoring, "word": 2, "quality": "correct", "score": 1.0,
             "withheld": None, "errors": [],
             "checks": [{**check, "status": _status(2)} for check in checks],
             "setting_checks": []} for scoring in ("strict", "shipped_default")]


class _Engine:
    """The pinned harness, standing in: one result per engine run of the request."""

    def __init__(self, kept: str = "correct") -> None:
        self.kept = kept
        self.requests: list[dict] = []

    def run(self, requests):
        self.requests.extend(requests)
        results = [{"item": request["item"], "scoring": engine["name"],
                    "final": [{"word": 2, "quality": self.kept,
                               "score": 1.0, "withheld": None, "errors": []}],
                    "letters": [],
                    "checks": [{"statuses": [_status(2), _status(3, "minor", 0.5)]},
                               {"statuses": [_status(2), _status(3)]}]}
                   for request in requests for engine in request["scorings"]]
        return {}, results


def _population() -> dict:
    return {ITEM: {"audio_sha256": SHA, "filename_label": "2:170"}}


def _trace_file(tmp_path, rows) -> "object":
    traces = tmp_path / "traces"
    traces.mkdir()
    with gzip.open(traces / f"{SHA}.jsonl.gz", "wt", encoding="utf-8") as sink:
        for row in rows:
            sink.write(json.dumps(row) + "\n")
    return traces


def _cells(tmp_path, rows, engine) -> dict:
    return _app_cells(_trace_file(tmp_path, rows), _population(), [(ITEM, 3)],
                      engine=engine, text=_Text(), log=lambda _message: None)


def test_a_reported_word_keeps_its_check_statuses_under_a_pending_row(tmp_path):
    cell = _cells(tmp_path, _saved_rows(), _Engine())[(ITEM, 3)]["strict"]
    assert (cell["quality"], cell["score"], cell["withheld"], cell["errors"]) == (
        "pending", None, None, [])
    assert [check["status"] for check in cell["checks"]] == [_status(3, "minor", 0.5),
                                                             _status(3)]
    assert cell["setting_checks"] == []


def test_a_check_that_graded_the_reported_word_keeps_it_off_m_nograde(tmp_path):
    cells = _cells(tmp_path, _saved_rows(), _Engine())
    arms = {"item": ITEM, "word": 3, "app": cells[(ITEM, 3)],
            "causes": {"strict": [], "shipped_default": []},
            "oracle": {"strict": _grade(), "shipped_default": _grade()},
            "teacher": {"strict": _grade(), "shipped_default": _grade()},
            "student": {"strict": None, "shipped_default": None}, "reference": None}
    result = attribute_listened(_key_row(), arms, {"verdict": "mistake"})
    assert (result["labels"], result["bucket"], result["side"]) == (
        ["M1u", "M3"], "M1u", FOLLOW_ALONG)
    rule = evaluate_rules([result])["rule_3"]
    assert rule["model_share"]["f_ref_follow_along"] == 0.0
    assert rule["unknown"]["words"] == 0


def test_a_replay_that_does_not_reproduce_the_gate_refuses_the_statuses(tmp_path):
    with pytest.raises(ValueError, match="does not reproduce the gate's"):
        _cells(tmp_path, _saved_rows(), _Engine(kept="minor"))


def test_a_clip_with_no_saved_check_is_left_without_an_app_cell(tmp_path):
    engine = _Engine()
    cells = _cells(tmp_path, [], engine)
    assert cells[(ITEM, 3)] == {"strict": None, "shipped_default": None}
    assert engine.requests == []


# --- R4.2: prereg §5's R-alone row ---------------------------------------------------------------
def test_the_reference_alone_is_reported_beside_the_unchanged_primary_bucket():
    result = attribute_listened(_key_row(), _arms(), {"verdict": "correct"})
    assert (result["bucket"], result["side"]) == ("F2", MODEL)
    assert result["alone"] is True
    rules = evaluate_rules([result])
    assert rules["rule_3"]["model_share"]["f_ref_follow_along"] == 1.0
    assert rules["r_alone"]["matches"] == 1
    assert rules["r_alone"]["moved"] == 1
    assert rules["r_alone"]["model_share"] == 0.0


def test_a_word_the_primary_already_reads_as_follow_along_is_not_moved():
    result = attribute_listened(_key_row(), _arms(_flagged(BEFORE + WORD)), {"verdict": "correct"})
    assert (result["bucket"], result["side"]) == ("F1-align", FOLLOW_ALONG)
    assert result["alone"] is True
    rules = evaluate_rules([result])
    assert rules["r_alone"] == {"label": rules["r_alone"]["label"], "matches": 1, "moved": 0,
                                "model_share": 0.0}
    assert rules["rule_3"]["model_share"]["f_ref_follow_along"] == 0.0


def test_an_ungraded_oracle_leaves_the_reference_alone_row_empty():
    result = attribute_listened(_key_row(), _arms(oracle=None), {"verdict": "correct"})
    assert (result["bucket"], result["alone"]) == ("F2", False)
    assert evaluate_rules([result])["r_alone"]["matches"] == 0


def test_a_reference_too_short_to_hold_by_nowhere_near_chance_is_not_alone(tmp_path):
    short = {"previous": BEFORE, "word": "بِ", "next": AFTER}
    result = attribute_listened(_key_row(), _arms(_flagged("بِ"), reference=short),
                                {"verdict": "correct"})
    assert (result["bucket"], result["alone"]) == ("F2", False)


# --- R4.3: unknown words alone still bound the share ---------------------------------------------
def _nograde(*, graded: bool, stratum: str = "S5", item: str = "n#0") -> dict:
    return {"item": item, "id": f"{item}", "outcome": "miss", "bucket": "M-nograde",
            "labels": ["M-nograde", "M3"], "teacher": "passed" if graded else "ungraded",
            "side": FOLLOW_ALONG if graded else UNKNOWN, "pi": 0.1, "stratum": stratum,
            "soft_or_target": False}


def test_unknown_words_alone_bound_the_share_at_zero_against_one():
    rule = evaluate_rules([_nograde(graded=False)])["rule_3"]
    assert rule["model_share"] == {"f_ref_follow_along": None, "f_ref_model": None}
    assert rule["weighted_without_s4"] is None
    assert rule["extremes"] == {"unknown_follow_along": 0.0, "unknown_model": 1.0}
    assert (rule["straddles_50"], rule["owner_decides"]) == (True, True)


def test_no_attributed_word_leaves_the_extremes_undefined():
    rule = evaluate_rules([])["rule_3"]
    assert rule["extremes"] == {"unknown_follow_along": None, "unknown_model": None}
    assert (rule["straddles_50"], rule["owner_decides"]) == (False, False)


# --- R4.4: a suspect explained by its own bucket keeps that bucket's side -------------------------
def test_a_suspect_explained_by_f1_stays_on_the_follow_along_side():
    result = attribute_listened(_key_row(), _arms(oracle="minor"), {"verdict": "correct"},
                                suspects=_suspects())
    assert (result["bucket"], result["side"], result["labels"]) == (
        "F1", FOLLOW_ALONG, ["F1"])


def test_a_suspect_explained_by_f1_align_stays_on_the_follow_along_side():
    result = attribute_listened(_key_row(), _arms(_flagged(BEFORE + WORD)),
                                {"verdict": "correct"}, suspects=_suspects())
    assert (result["bucket"], result["side"]) == ("F1-align", FOLLOW_ALONG)


def test_the_both_sides_marker_is_the_f_ref_buckets_alone():
    result = attribute_listened(_key_row(), _arms(), {"verdict": "correct"},
                                suspects=_suspects())
    assert (result["bucket"], result["side"]) == ("F-ref", "f-ref")

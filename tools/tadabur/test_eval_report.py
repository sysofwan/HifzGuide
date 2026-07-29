"""Tests for the pure two-sided eval core (issue #7): confusion matrix + the
should-accept recall / should-reject discrimination metrics, on hand-built decodes."""

from __future__ import annotations

import json

from tadabur.eval_fixtures import ACCEPT, REJECT
from tadabur.eval_report import (
    OTHER,
    STRICT_THRESHOLD,
    SUCCESS_CRITERION,
    ClipDecode,
    SideMetrics,
    evaluate,
    strict_accept,
)

# Soft pair ذ↔ز (codepoint-ordered label). Reference strings are the normalized cache
# form (no diacritics, shadda doubled), exactly what the gate compares against.
DHAL, ZAY, BA, NOON = "\u0630", "\u0632", "\u0628", "\u0646"
CONTRAST_DZ = f"{DHAL}\u2194{ZAY}"


def _accept(clip_id: str, predicted: str, reference: str, contrast: str = CONTRAST_DZ) -> ClipDecode:
    return ClipDecode(clip_id, contrast, ACCEPT, predicted, reference)


def _reject(clip_id: str, predicted: str, reference: str, contrast: str = CONTRAST_DZ) -> ClipDecode:
    return ClipDecode(clip_id, contrast, REJECT, predicted, reference)


def test_side_metrics_ratios_and_empty():
    side = SideMetrics(total=4, accepted=3)
    assert side.rejected == 1
    assert side.recall == 0.75
    assert side.discrimination == 0.25
    empty = SideMetrics(total=0, accepted=0)
    assert empty.recall is None and empty.discrimination is None


def test_strict_accept_identical_decode_passes():
    # An exact decode scores match_ratio 1.0 >= the strict threshold.
    assert strict_accept(_accept("a", f"{ZAY}{BA}{NOON}", f"{ZAY}{BA}{NOON}"))


def test_strict_rejects_soft_pair_substitution():
    # A decode that clears match_ratio but substitutes a soft-pair partner (ذ for ز)
    # is a hard mismatch under .strict (soft pairs off), so it is rejected.
    ref = f"{BA}{ZAY}{BA}{NOON}"
    sub = f"{BA}{DHAL}{BA}{NOON}"
    assert not strict_accept(_accept("a", sub, ref))
    assert strict_accept(_accept("a", ref, ref))


def test_strict_accept_garbage_decode_fails():
    assert not strict_accept(_reject("a", "\u0643\u0645\u0644", f"{ZAY}{BA}{NOON}"))


def test_recall_and_discrimination():
    ref = f"{BA}{ZAY}{BA}{NOON}"
    clips = [
        _accept("a1", ref, ref),                       # exact -> admitted (recall)
        _accept("a2", "\u0643\u0645\u0644\u0642", ref),  # wrong -> false negative
        _reject("r1", "\u0643\u0645\u0644\u0642", ref),  # wrong -> still rejected (good)
        _reject("r2", ref, ref),                        # admitted a reject -> discrimination lost
    ]
    report = evaluate(clips, "test-model")
    assert report.should_accept.total == 2 and report.should_accept.accepted == 1
    assert report.should_accept.recall == 0.5
    assert report.should_reject.total == 2 and report.should_reject.rejected == 1
    assert report.should_reject.discrimination == 0.5


def test_soft_pair_confusion_counts_substitution():
    # Reference has ز; the decode renders it as its confusable partner ذ.
    clip = _accept("a", f"{BA}{DHAL}{BA}", f"{BA}{ZAY}{BA}")
    report = evaluate([clip], "m")
    conf = {c.contrast: c.matrix for c in report.soft_pair_confusion}
    assert CONTRAST_DZ in conf
    matrix = conf[CONTRAST_DZ]
    assert matrix[ZAY][DHAL] == 1          # ref ز rendered as partner ذ
    assert matrix[ZAY][ZAY] == 0
    assert matrix[ZAY][OTHER] == 0
    # Every soft pair is present with fully-populated rows/cols for comparability.
    for other_matrix in conf.values():
        for row in other_matrix.values():
            assert set(row) >= {OTHER}


def test_soft_pair_confusion_correct_and_other():
    correct = _accept("a", f"{BA}{ZAY}{BA}", f"{BA}{ZAY}{BA}")
    report = evaluate([correct], "m")
    matrix = {c.contrast: c.matrix for c in report.soft_pair_confusion}[CONTRAST_DZ]
    assert matrix[ZAY][ZAY] == 1


def test_shadda_confusion_dropped_and_added():
    # Reference is used verbatim, so a bare doubled core is a geminate there; the decode
    # is normalized, so its added gemination must survive normalization — a following
    # harakat (fatha) breaks the bare run and keeps the doubling (see normalization).
    fatha = "\u064e"
    dropped = _accept("d", f"{BA}{NOON}{BA}", f"{BA}{NOON}{NOON}{BA}")            # ref geminate, decode single
    added = _reject("a", f"{BA}{NOON}{NOON}{fatha}{BA}", f"{BA}{NOON}{BA}")       # decode doubled
    report = evaluate([dropped, added], "m")
    assert report.shadda_confusion.dropped == 1
    assert report.shadda_confusion.added == 1


def test_per_contrast_breakdown():
    ref = f"{BA}{ZAY}{BA}{NOON}"
    clips = [
        _accept("a", ref, ref, contrast=CONTRAST_DZ),
        _reject("r", ref, ref, contrast="shadda"),
    ]
    report = evaluate(clips, "m")
    assert report.per_contrast[CONTRAST_DZ]["should_accept"].total == 1
    assert report.per_contrast[CONTRAST_DZ]["should_reject"].total == 0
    assert report.per_contrast["shadda"]["should_reject"].total == 1


def test_report_json_is_deterministic_and_complete():
    ref = f"{BA}{ZAY}{BA}{NOON}"
    clips = [_accept("a", ref, ref), _reject("r", "\u0643\u0645\u0644\u0642", ref)]
    a = evaluate(clips, "m").to_json_dict()
    b = evaluate(clips, "m").to_json_dict()
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)
    assert a["strict_threshold"] == STRICT_THRESHOLD
    assert a["success_criterion"] == SUCCESS_CRITERION
    assert a["should_accept"]["recall"] == 1.0
    assert a["should_reject"]["discrimination"] == 1.0
    assert CONTRAST_DZ in a["confusion_matrix"]["soft_pairs"]
    assert set(a["confusion_matrix"]["shadda"]) == {"added", "dropped"}


def test_empty_clips_yield_null_headline_metrics():
    report = evaluate([], "m")
    j = report.to_json_dict()
    assert j["should_accept"]["recall"] is None
    assert j["should_reject"]["discrimination"] is None
    # Confusion matrix still lists every soft pair with zeroed cells.
    assert len(j["confusion_matrix"]["soft_pairs"]) == 6


def test_clip_outcomes_expose_disagreement_that_equal_counts_hide():
    """Why per-clip outcomes exist: the aggregate counts are not a paired comparison.

    Two checkpoints can post an identical ``accepted`` total while disagreeing on
    individual clips in both directions. Comparing only the totals reports "no change";
    McNemar over the discordant clips is the correct paired test, and it needs these rows.
    """
    good, bad = f"{ZAY}{BA}{NOON}", f"{DHAL}{BA}{NOON}"
    rung_a = [_reject("c1", good, good), _reject("c2", bad, good)]
    rung_b = [_reject("c1", bad, good), _reject("c2", good, good)]

    report_a, report_b = evaluate(rung_a, "a"), evaluate(rung_b, "b")
    assert report_a.should_reject.accepted == report_b.should_reject.accepted

    def by_clip(report):
        return {o.clip_id: o.accepted for o in report.clip_outcomes}

    discordant = [c for c in by_clip(report_a) if by_clip(report_a)[c] != by_clip(report_b)[c]]
    assert sorted(discordant) == ["c1", "c2"]


def test_clip_outcomes_serialize_deterministically_by_clip_id():
    clips = [_accept("z", f"{ZAY}{BA}", f"{ZAY}{BA}"), _accept("a", f"{DHAL}{BA}", f"{ZAY}{BA}")]
    payload = evaluate(clips, "m").to_json_dict()["clip_outcomes"]
    assert [row["clip_id"] for row in payload] == ["a", "z"]
    assert payload[0] == {"clip_id": "a", "contrast": CONTRAST_DZ, "verdict": ACCEPT,
                          "accepted": False}
    assert json.dumps(payload)

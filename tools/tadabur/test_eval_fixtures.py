"""Tests for the eval fixture schema + loader (issue #16): the shared contract
the P3.5 audit (#6) populates and the P3.7 eval (#7) consumes."""

from __future__ import annotations

import json

import pytest

from tadabur.eval_fixtures import (
    ACCEPT,
    REJECT,
    SCHEMA_FIELDS,
    SHOULD_ACCEPT_PATH,
    SHOULD_REJECT_PATH,
    EvalFixtureEntry,
    load_eval_fixtures,
    load_should_accept,
    load_should_reject,
)


def _write(path, entries: list[dict]) -> None:
    path.write_text(
        "\n".join(json.dumps(e, ensure_ascii=False) for e in entries) + "\n",
        encoding="utf-8",
    )


def _entry(clip_id: str, verdict: str, contrast: str = "\u0633\u2194\u0635") -> dict:
    return {
        "clip_id": clip_id,
        "audio_ref": f"{clip_id}.wav",
        "surah_ayah": "2:255",
        "contrast": contrast,
        "verdict": verdict,
    }


def test_schema_fields_match_dataclass():
    assert SCHEMA_FIELDS == ("clip_id", "audio_ref", "surah_ayah", "contrast", "verdict", "note")


def test_shipped_sets_load_cleanly():
    # The committed fixture files carry the P3.5 poison-audit verdicts; whatever they
    # hold, the loaders must parse every line into a well-formed, correctly-labelled entry.
    assert SHOULD_ACCEPT_PATH.exists()
    assert SHOULD_REJECT_PATH.exists()
    accept = load_should_accept()
    reject = load_should_reject()
    assert all(e.verdict == ACCEPT for e in accept)
    assert all(e.verdict == REJECT for e in reject)


def test_missing_file_yields_no_entries(tmp_path):
    assert load_eval_fixtures(tmp_path / "nope.jsonl", ACCEPT) == []


def test_round_trip_load(tmp_path):
    path = tmp_path / "should_accept.jsonl"
    _write(path, [_entry("a", ACCEPT), _entry("b", ACCEPT, contrast="shadda")])
    entries = load_eval_fixtures(path, ACCEPT)
    assert entries == [
        EvalFixtureEntry("a", "a.wav", "2:255", "\u0633\u2194\u0635", ACCEPT),
        EvalFixtureEntry("b", "b.wav", "2:255", "shadda", ACCEPT),
    ]


def test_blank_and_comment_lines_ignored(tmp_path):
    path = tmp_path / "s.jsonl"
    path.write_text(
        "# a header comment\n\n" + json.dumps(_entry("a", REJECT)) + "\n",
        encoding="utf-8",
    )
    assert [e.clip_id for e in load_eval_fixtures(path, REJECT)] == ["a"]


def test_verdict_mismatch_rejected(tmp_path):
    path = tmp_path / "should_accept.jsonl"
    _write(path, [_entry("a", REJECT)])  # reject entry in the accept set
    with pytest.raises(ValueError, match="verdict"):
        load_eval_fixtures(path, ACCEPT)


def test_unknown_contrast_rejected(tmp_path):
    path = tmp_path / "s.jsonl"
    _write(path, [_entry("a", ACCEPT, contrast="\u062B\u2194\u0642")])  # not a real bucket
    with pytest.raises(ValueError, match="contrast"):
        load_eval_fixtures(path, ACCEPT)


def test_unknown_field_rejected(tmp_path):
    path = tmp_path / "s.jsonl"
    bad = _entry("a", ACCEPT)
    bad["oops"] = 1
    _write(path, [bad])
    with pytest.raises(ValueError, match="unknown fixture field"):
        load_eval_fixtures(path, ACCEPT)


def test_marginal_contrast_is_valid(tmp_path):
    path = tmp_path / "s.jsonl"
    _write(path, [_entry("a", REJECT, contrast="marginal")])
    assert load_eval_fixtures(path, REJECT)[0].contrast == "marginal"


def test_bad_expected_verdict_argument():
    with pytest.raises(ValueError, match="expected_verdict"):
        load_eval_fixtures(SHOULD_ACCEPT_PATH, "maybe")

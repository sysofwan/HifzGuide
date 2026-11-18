"""One protocol for every teacher-agreement number, and the comparison a card publishes.

The card's table and the CoreML export's table must be the same measurement, so both call the
helpers here: this test pins the aggregation (accuracy against the teacher, drift against the
first model) and the streamed decode, and checks that a model directory is scored through the
same loop a checkpoint-backed export uses.
"""

from __future__ import annotations

import numpy as np
import pytest

from training import hub_agreement
from training.decode_evalset import SCHEMA_VERSION, EvalClip, EvalSet
from training.decoding import PROTOCOL_VERSION
from training.distill_loss import CONFIRM_TIMESTEPS


def _clip(name: str, teacher: str, reciter: int = 1):
    return EvalClip(
        filename=name,
        surah_ayah="78:1",
        reciter_id=reciter,
        shard=20,
        duration_s=4.0,
        split="dev",
        teacher_text=teacher,
    )


def _evalset(clips, **provenance):
    base = {
        "teacher_model_id": hub_agreement.DEFAULT_EXTRACTOR,
        "confirm_timesteps": CONFIRM_TIMESTEPS,
        "protocol_version": PROTOCOL_VERSION,
    }
    base.update(provenance)
    return EvalSet(
        schema_version=SCHEMA_VERSION,
        clips=tuple(clips),
        num_scanned=len(clips),
        num_skipped=0,
        provenance=base,
    )


def test_agreement_rows_scores_each_model_against_the_teacher_and_the_first_model():
    clips = [_clip("a.flac", "ببت"), _clip("b.flac", "تث")]
    decodes = {
        "reference": {"a.flac": "ببت", "b.flac": "تث"},
        "other": {"a.flac": "ببج", "b.flac": "تث"},
    }

    rows = hub_agreement.agreement_rows(clips, decodes)

    assert [row.model for row in rows] == ["reference", "other"]
    assert rows[0].char_accuracy == 1.0
    assert rows[0].drift_from_reference == 0.0
    assert rows[1].char_accuracy == pytest.approx(4 / 5)
    # One character of five moved, whether toward the teacher or away: drift is the upper bound.
    assert rows[1].drift_from_reference == pytest.approx(1 / 5)
    assert rows[1].ci_low <= rows[1].char_accuracy <= rows[1].ci_high


def test_decode_windows_streams_the_frames_and_collapses_the_ctc_path():
    from tadabur.phoneme_vocab import PHONEME_ID_TO_CHAR

    ids = [3, 3, 4, 0, 5]

    def logits_for_window(_features):
        logits = np.full((hub_agreement.DEPLOYED_LOGIT_FRAMES + 10, 43), -10.0, dtype="float32")
        for frame, token in enumerate(ids):
            logits[frame, token] = 10.0
        return logits

    decoded = hub_agreement.decode_windows(logits_for_window, [np.zeros((250, 160))])

    assert decoded == "".join(PHONEME_ID_TO_CHAR[i] for i in (3, 4, 5))


def test_score_runs_every_model_over_the_same_clips(monkeypatch, tmp_path):
    clips = [_clip("a.flac", "ببت", reciter=1), _clip("b.flac", "تث", reciter=2)]
    monkeypatch.setattr(hub_agreement, "load_manifest", lambda _path: _evalset(clips))
    monkeypatch.setattr(hub_agreement, "read_clip_audio", lambda _dir, _name: np.zeros(16000))
    monkeypatch.setattr(
        hub_agreement, "window_features", lambda _extractor, _samples: [np.zeros((250, 160))]
    )

    ids_by_label = {"reference": [3, 3, 4, 0, 5], "other": [3, 3, 4, 0]}

    def scorer(directory):
        label = directory.name

        def logits_for_window(_features):
            logits = np.full((125, 43), -10.0, dtype="float32")
            for frame, token in enumerate(ids_by_label[label]):
                logits[frame, token] = 10.0
            return logits

        return logits_for_window

    rows = hub_agreement.score(
        tmp_path,
        {"reference": tmp_path / "reference", "other": tmp_path / "other"},
        num_clips=10,
        extractor=object(),
        scorer=scorer,
    )

    assert [row.model for row in rows] == ["reference", "other"]
    assert rows[0].drift_from_reference == 0.0
    assert rows[1].drift_from_reference > 0


def test_score_refuses_a_manifest_built_under_other_rules(monkeypatch, tmp_path):
    """A comparison table is only a measurement against the truth it names."""
    clips = [_clip("a.flac", "ببت"), _clip("b.flac", "تث")]
    stale = _evalset(clips, protocol_version="obsolete")
    monkeypatch.setattr(hub_agreement, "load_manifest", lambda _path: stale)

    def not_called(_directory):
        raise AssertionError("scoring began before the manifest was checked")

    with pytest.raises(SystemExit, match="protocol_version"):
        hub_agreement.score(tmp_path, {"reference": tmp_path}, num_clips=2, scorer=not_called)

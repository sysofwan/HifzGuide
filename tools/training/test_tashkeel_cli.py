"""The haraka tools' command-line paths, end to end through what they write.

Each ``main()`` decodes, classifies and serialises; a bug in how the decode's fingerprint
travels to the output only shows up once something is written. These run the real CLIs on a
real label file, with the model and the clip audio replaced by stand-ins -- no GPU, no
download -- and read back what landed on disk.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import pytest
import torch

from training import decoding as dc
from training import tashkeel_outcomes, tashkeel_worklist, windowed_batch
from training.tashkeel_eval import FATHA, KASRA, MATCHED, OMITTED
from training.windowed_labels import WindowLabel, write_labels

REFERENCES = [f"م{FATHA}الك{KASRA}", f"ب{KASRA}سم{FATHA}"]
# What each model hears in each window: the base drops the kasra of window 0, the
# candidate gets every vowel.
SCRIPT = {
    "base-model": [f"م{FATHA}الك", f"ب{KASRA}سم{FATHA}"],
    "candidate-model": [REFERENCES[0], REFERENCES[1]],
}


class _ScriptedDecoder(dc.Decoder):
    """A real :class:`Decoder` -- so its fingerprint is the real one -- with scripted spans."""

    def __init__(self, model_ref: str, batch_size: int):
        super().__init__(model_ref, torch.nn.Linear(1, 1), None, "cpu", batch_size)

    def decode_spans(self, spans):
        return [SCRIPT[self.model_ref][i] for i, _ in enumerate(spans)]


class _SilentClips:
    def __init__(self, audio_dir):
        pass

    def waveform(self, clip_audio_filename):
        return np.zeros(2 * dc.SAMPLE_RATE, dtype=np.float32)


@pytest.fixture
def cli(tmp_path, monkeypatch):
    """Labels on disk, a scripted model behind ``Decoder.load`` and silent clip audio."""
    labels = tmp_path / "labels.jsonl"
    write_labels(
        labels,
        [
            WindowLabel(
                clip_audio_filename=f"clip{i}.wav", surah_ayah="1:1", reciter_id=i,
                window_index=0, start_sample=0, num_samples=dc.SAMPLE_RATE,
                recitation_start_sample=0, feature_frames=50, logit_frames=25,
                phoneme_label=reference, word_start=0, word_end=1, segment_indices=(0,),
            )
            for i, reference in enumerate(REFERENCES)
        ],
        "val",
    )
    monkeypatch.setattr(
        dc.Decoder,
        "load",
        classmethod(lambda cls, ref, device, *, weights_dtype, batch_size=16: (
            _ScriptedDecoder(str(ref), batch_size)
        )),
    )
    monkeypatch.setattr(windowed_batch, "ClipAudioCache", _SilentClips)

    def run(module, *argv):
        monkeypatch.setattr(sys, "argv", [module.__name__, *map(str, argv)])
        module.main()

    common = ("--labels", labels, "--audio-dir", tmp_path, "--device", "cpu")
    return tmp_path, run, common


def _expected_fingerprint(model_ref: str) -> dict:
    return _ScriptedDecoder(model_ref, 8).fingerprint(dc.SPANS).as_dict()


def test_static_mining_writes_the_worklist_and_a_sidecar_naming_the_base_decode(cli):
    tmp_path, run, common = cli
    out = tmp_path / "static.jsonl"
    run(tashkeel_worklist, *common, "--base", "base-model", "--out", out)

    summary = json.loads((tmp_path / "static.jsonl.summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "static"
    assert summary["base_decode"] == _expected_fingerprint("base-model")
    assert summary["candidate_decode"] is None
    sites = tashkeel_worklist.read_worklist(out)
    assert sites and {s.base_outcome for s in sites} == {MATCHED, OMITTED}


def test_paired_mining_records_both_decodes(cli):
    tmp_path, run, common = cli
    out = tmp_path / "paired.jsonl"
    run(
        tashkeel_worklist, *common,
        "--base", "base-model", "--candidate", "candidate-model", "--out", out,
    )

    summary = json.loads((tmp_path / "paired.jsonl.summary.json").read_text(encoding="utf-8"))
    assert summary["mode"] == "paired"
    assert summary["base_decode"] == _expected_fingerprint("base-model")
    assert summary["candidate_decode"] == _expected_fingerprint("candidate-model")
    (recovered,) = tashkeel_worklist.read_worklist(out)  # the one kasra base dropped
    assert (recovered.base_outcome, recovered.candidate_outcome) == (OMITTED, MATCHED)


def test_scoring_a_candidate_writes_its_fingerprint_header_and_every_outcome(cli):
    tmp_path, run, common = cli
    worklist = tmp_path / "static.jsonl"
    run(tashkeel_worklist, *common, "--base", "base-model", "--out", worklist)
    out = tmp_path / "outcomes.jsonl"
    run(
        tashkeel_outcomes, *common,
        "--worklist", worklist, "--model", "candidate-model", "--out", out,
    )

    fingerprint, outcomes = tashkeel_outcomes.read_outcomes(out)
    assert fingerprint.as_dict() == _expected_fingerprint("candidate-model")
    sites = tashkeel_worklist.read_worklist(worklist)
    assert set(outcomes) == {s.site_id for s in sites}
    assert {o.outcome for o in outcomes.values()} == {MATCHED}
    # The worklist's base decode and this candidate's are comparable: only the model differs.
    summary = json.loads((tmp_path / "static.jsonl.summary.json").read_text(encoding="utf-8"))
    dc.DecodeFingerprint.from_dict(summary["base_decode"], "summary").check_comparable(
        fingerprint, "base and candidate"
    )


def test_a_failed_scoring_run_leaves_the_previous_outcomes_intact(cli):
    tmp_path, run, common = cli
    worklist = tmp_path / "static.jsonl"
    run(tashkeel_worklist, *common, "--base", "base-model", "--out", worklist)
    out = tmp_path / "outcomes.jsonl"
    out.write_text("previous run\n", encoding="utf-8")

    # A worklist naming a window the labels do not hold fails before anything is written.
    sites = tashkeel_worklist.read_worklist(worklist)
    stray = type(sites[0])(**{**sites[0].__dict__, "clip_audio_filename": "elsewhere.wav"})
    tashkeel_worklist.write_worklist(worklist, [*sites, stray])
    with pytest.raises(ValueError, match="missing"):
        run(
            tashkeel_outcomes, *common,
            "--worklist", worklist, "--model", "candidate-model", "--out", out,
        )
    assert out.read_text(encoding="utf-8") == "previous run\n"

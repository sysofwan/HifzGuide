"""The hub stage carries the weights the round scored, and a config that can be loaded back.

A staged directory is only useful if a reader can rebuild the module from it without asking
which weight set it holds, so the state dict is checked by loading it into a fresh student
rather than by comparing shapes, and the transcribed config is checked against the preset it
names. The refusal cases matter as much: merging into an old stage mixes two weight sets, and
an ``--ema`` stage of a live-only checkpoint would publish the wrong model under the averaged
name.

The fixture is an ``h256`` checkpoint -- the point is selection and transcription, not width.
"""

from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

import stage_hf_model
from training.distill_student import PRESETS, build_student
from training.distill_train import TrainConfig, WeightAverage, save_checkpoint


@pytest.fixture(scope="module")
def candidate(tmp_path_factory):
    """A checkpoint carrying both weight sets, as a run with ``--ema-decay`` writes one."""
    directory = tmp_path_factory.mktemp("checkpoint")
    path = directory / "checkpoint.pt"

    student = build_student(PRESETS["h256"])
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.zero_()
    averager = WeightAverage(student, 0.5)
    with torch.no_grad():
        for parameter in student.parameters():
            parameter.fill_(0.25)
    averager.update(student)

    optimizer = torch.optim.SGD(student.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    save_checkpoint(
        path, 10, student, student, optimizer, scheduler,
        TrainConfig(preset="h256", audio_root="", out_dir=str(directory), ema_decay=0.5),
        averager,
    )

    processor = directory / "preprocessor_config.json"
    processor.write_text('{"sampling_rate": 16000, "stride": 2}\n', encoding="utf-8")

    state = torch.load(path, map_location="cpu", weights_only=False)
    return path, state["ema_state"]["shadow"], processor


def test_the_stage_holds_the_averaged_weights_a_config_that_loads_them_and_a_vocab(
    candidate, tmp_path
):
    path, shadow, processor = candidate
    out = tmp_path / "stage"

    checksums = stage_hf_model.stage(path, use_ema=True, processor_config=processor, output_dir=out)

    config = json.loads((out / "config.json").read_text(encoding="utf-8"))
    assert config["weights"] == "ema"
    assert config["preset"] == "h256"
    assert config["hidden_size"] == PRESETS["h256"].hidden_size
    assert config["num_attention_heads"] == PRESETS["h256"].num_attention_heads
    assert config["step"] == 10
    assert config["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert config["feature_input"]["shape"] == [250, 160]
    assert config["logit_frames"] == 125
    assert config["phoneme_classes"] == 43

    staged = torch.load(out / "student_state_dict.pt", map_location="cpu", weights_only=True)
    student = build_student(PRESETS["h256"])
    student.load_state_dict(staged)
    loaded = student.state_dict()
    for name, tensor in shadow.items():
        assert torch.allclose(loaded[name], tensor.to(loaded[name].dtype)), name

    vocab = json.loads((out / "phoneme_vocab.json").read_text(encoding="utf-8"))
    from tadabur.phoneme_vocab import PHONEME_ID_TO_CHAR

    assert vocab["classes"] == 43 == len(vocab["id_to_char"])
    assert vocab["id_to_char"] == list(PHONEME_ID_TO_CHAR)

    for name, digest in checksums.items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest
    assert (out / stage_hf_model.CHECKSUMS_NAME).read_text(encoding="utf-8") == "".join(
        f"{digest}  {name}\n" for name, digest in checksums.items()
    )


def test_a_non_empty_output_directory_is_refused(candidate, tmp_path):
    """Writing over an older stage would mix two weight sets under one checksum table."""
    path, _, processor = candidate
    out = tmp_path / "stage"
    out.mkdir()
    (out / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="not empty"):
        stage_hf_model.stage(path, use_ema=True, processor_config=processor, output_dir=out)


def test_staging_ema_weights_from_a_live_only_checkpoint_fails_loudly(candidate, tmp_path):
    path, _, processor = candidate
    state = torch.load(path, map_location="cpu", weights_only=False)
    del state["ema_state"]
    live_only = tmp_path / "live_only.pt"
    torch.save(state, live_only)

    with pytest.raises(SystemExit):
        stage_hf_model.stage(
            live_only, use_ema=True, processor_config=processor, output_dir=tmp_path / "stage"
        )
    assert not (tmp_path / "stage").exists()

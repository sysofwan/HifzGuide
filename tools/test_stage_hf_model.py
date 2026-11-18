"""The stage loads the way the published repository does, and holds the weights the round scored.

"Loads the same way" is the whole point of matching the published layout, so it is checked
through the consumer's own entry points -- `AutoConfig`/`AutoModel` with `trust_remote_code`
against the staged directory -- rather than by reading the config JSON back as a plain dict.
The weight check loads `model.safetensors` and compares it with the averaged shadow, because a
valid package of the *live* weights would pass every shape and size assertion.

The fixture is an `h256` checkpoint -- the point is layout and selection, not width.
"""

from __future__ import annotations

import hashlib
import json

import pytest

torch = pytest.importorskip("torch")

import stage_hf_model
from training.distill_student import PRESETS, TEACHER_MODEL_ID, build_student
from training.distill_train import TrainConfig, WeightAverage, save_checkpoint

EXPECTED_FILES = {
    "model.safetensors",
    "config.json",
    "configuration_multi_level_ctc.py",
    "modeling_multi_level_ctc.py",
    "phoneme_vocab.json",
    "preprocessor_config.json",
}


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

    state = torch.load(path, map_location="cpu", weights_only=False)
    return path, state["student"], state["ema_state"]["shadow"]


def test_the_stage_is_the_published_file_set_and_loads_through_trust_remote_code(
    candidate, tmp_path
):
    path, live, shadow = candidate
    out = tmp_path / "stage"

    checksums = stage_hf_model.stage(path, use_ema=True, output_dir=out, teacher_revision=None)

    assert set(checksums) == EXPECTED_FILES
    assert {p.name for p in out.iterdir()} == EXPECTED_FILES

    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(out, trust_remote_code=True)
    assert config.architectures == ["Wav2Vec2BertForMultilevelCTC"]
    assert config.auto_map == stage_hf_model.AUTO_MAP
    assert config.hidden_size == PRESETS["h256"].hidden_size
    assert config.num_hidden_layers == PRESETS["h256"].num_hidden_layers
    assert config.level_to_vocab_size["phonemes"] == 43
    assert config.pad_token_id == 0
    assert len(config.id2label) == 43 and config.id2label[0] == "[PAD]"
    assert config.label2id["[PAD]"] == 0

    model = AutoModel.from_pretrained(out, trust_remote_code=True)
    staged = model.state_dict()
    for name, tensor in shadow.items():
        assert torch.equal(staged[name].to(tensor.dtype), tensor), name
    # A stage of the live weights would satisfy every shape and layout assertion, so assert
    # the two sets really differ and that the stage followed the averaged one.
    assert any(not torch.equal(staged[name].float(), live[name].float()) for name in shadow)

    for name, digest in checksums.items():
        assert hashlib.sha256((out / name).read_bytes()).hexdigest() == digest


def test_the_vocab_and_processor_match_what_a_reader_decodes_with(candidate, tmp_path):
    path, _, _ = candidate
    out = tmp_path / "stage"
    stage_hf_model.stage(path, use_ema=True, output_dir=out, teacher_revision=None)

    from tadabur.phoneme_vocab import PHONEME_ID_TO_CHAR

    vocab = json.loads((out / "phoneme_vocab.json").read_text(encoding="utf-8"))
    assert vocab["id_to_char"] == list(PHONEME_ID_TO_CHAR)
    assert vocab["pad_id"] == 0 and vocab["pad_token"] == "[PAD]"

    processor = json.loads((out / "preprocessor_config.json").read_text(encoding="utf-8"))
    assert processor["feature_extractor_type"] == "SeamlessM4TFeatureExtractor"
    assert processor["sampling_rate"] == 16000 and processor["stride"] == 2
    assert processor["padding_value"] == 1
    assert "processor_class" not in processor
    assert TEACHER_MODEL_ID.endswith("muaalem-model-v3_2")


def test_a_non_empty_output_directory_is_refused(candidate, tmp_path):
    """Writing over an older stage would mix two weight sets under one repository."""
    path, _, _ = candidate
    out = tmp_path / "stage"
    out.mkdir()
    (out / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="not empty"):
        stage_hf_model.stage(path, use_ema=True, output_dir=out, teacher_revision=None)


def test_staging_ema_weights_from_a_live_only_checkpoint_fails_loudly(candidate, tmp_path):
    path, _, _ = candidate
    state = torch.load(path, map_location="cpu", weights_only=False)
    del state["ema_state"]
    live_only = tmp_path / "live_only.pt"
    torch.save(state, live_only)

    with pytest.raises(SystemExit):
        stage_hf_model.stage(
            live_only, use_ema=True, output_dir=tmp_path / "stage", teacher_revision=None
        )
    assert not (tmp_path / "stage").exists()

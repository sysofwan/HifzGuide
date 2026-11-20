"""The hub publication is one tool, and it publishes the weights the round scored.

`publish_hf` produced the currently published `sysofwan/hifzguide-muaalem-mini`; a second
exporter beside it would let the next release publish whichever weight set its author
remembered. So the layout, the shape contract, the weight-set choice and the teacher pin are
pinned here, through the consumer's own entry points (`AutoConfig`/`AutoModel` with
`trust_remote_code`) rather than by reading the files back as dicts.

The fixture is an `h256` checkpoint -- the point is selection, pinning and layout, not width --
and the feature extractor is stubbed, so the suite needs no hub access.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

torch = pytest.importorskip("torch")

from training import publish_hf
from training.distill_student import PRESETS, build_student
from training.distill_train import TrainConfig, WeightAverage, save_checkpoint

TEACHER_REVISION = "01a1ef9fbe40d144ef845101e89ff924aed3fef5"
#: The revision every QUD read used; the notice names it, so the credit ages with the read.
QUD_SOURCES = Path(publish_hf.__file__).parent.parent / "tadabur" / "qud_eval" / "sources.json"
EXPECTED_FILES = {
    ".gitattributes",
    "LICENSE",
    "NOTICE.md",
    "README.md",
    "config.json",
    "configuration_multi_level_ctc.py",
    "model.safetensors",
    "modeling_multi_level_ctc.py",
    "phoneme_vocab.json",
    "preprocessor_config.json",
}


class _StubExtractor:
    """Stands in for the teacher's extractor: the stage only needs a file to save."""

    def save_pretrained(self, directory) -> None:
        Path(directory, "preprocessor_config.json").write_text(
            '{"feature_extractor_type": "Stub", "sampling_rate": 16000}\n', encoding="utf-8"
        )


@pytest.fixture
def extractor_calls(monkeypatch):
    """Record the `from_pretrained` kwargs the stage asks the hub for."""
    calls: list[dict] = []

    def from_pretrained(model_id, **kwargs):
        calls.append({"model_id": model_id, **kwargs})
        return _StubExtractor()

    monkeypatch.setattr(
        publish_hf.SeamlessM4TFeatureExtractor, "from_pretrained", staticmethod(from_pretrained)
    )
    return calls


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
        TrainConfig(
            preset="h256",
            audio_root="",
            out_dir=str(directory),
            ema_decay=0.5,
            teacher_revision=TEACHER_REVISION,
            augment={"version": "aug94-mixture-v2"},
        ),
        averager,
    )

    state = torch.load(path, map_location="cpu", weights_only=False)
    return path, state["student"], state["ema_state"]["shadow"]


@pytest.fixture
def release_identity(candidate, monkeypatch):
    """Stand the synthetic checkpoint in for the artifact the NC card reports.

    The card's measured claims are bound to one checkpoint hash and its ema weights; the
    identity check itself is what the rejection tests below exercise.
    """
    path, _, _ = candidate
    monkeypatch.setattr(
        publish_hf,
        "RELEASE_CANDIDATE",
        {"checkpoint_sha256": publish_hf.sha256_file(path), "weights": "ema"},
    )


def test_publishing_a_student_stages_the_published_layout_and_the_ema_weights(
    candidate, extractor_calls, release_identity, tmp_path
):
    path, live, shadow = candidate
    out = tmp_path / "mini-v2"

    record = publish_hf.export_and_publish(
        checkpoint_path=path,
        out_dir=out,
        repo_id="sysofwan/hifzguide-muaalem-mini-v2",
        profile="cc-by-nc",
        use_ema=True,
    )

    assert {p.name for p in out.iterdir()} == EXPECTED_FILES
    assert set(record["sha256"]) == EXPECTED_FILES
    assert record["weights"] == "ema"
    assert record["step"] == 10
    assert record["shape"] == [125, 43]
    assert record["teacher_revision"] == TEACHER_REVISION
    assert record["checkpoint_sha256"] == publish_hf.sha256_file(path)
    for name, digest in record["sha256"].items():
        assert publish_hf.sha256_file(out / name) == digest

    from transformers import AutoConfig, AutoModel

    config = AutoConfig.from_pretrained(out, trust_remote_code=True)
    assert config.architectures == ["Wav2Vec2BertForMultilevelCTC"]
    assert config.auto_map == {
        "AutoConfig": "configuration_multi_level_ctc.Wav2Vec2BertForMultilevelCTCConfig",
        "AutoModel": "modeling_multi_level_ctc.Wav2Vec2BertForMultilevelCTC",
    }
    assert config.hidden_size == PRESETS["h256"].hidden_size
    assert len(config.id2label) == 43 and config.id2label[0] == "[PAD]"

    staged = AutoModel.from_pretrained(out, trust_remote_code=True).state_dict()
    for name, tensor in shadow.items():
        assert torch.equal(staged[name].to(tensor.dtype), tensor), name
    # A publication of the live weights would satisfy every layout and shape assertion, so
    # assert the two sets differ and the stage followed the averaged one.
    assert any(not torch.equal(staged[name].float(), live[name].float()) for name in shadow)

    card = (out / "README.md").read_text(encoding="utf-8")
    notice = (out / "NOTICE.md").read_text(encoding="utf-8")
    front_matter = yaml.safe_load(card.split("---")[1])
    assert front_matter["license"] == "cc-by-nc-4.0"
    assert front_matter["datasets"] == ["FaisaI/tadabur", "tarteel-ai/tlog"]
    assert (out / "LICENSE").read_bytes() == publish_hf.CC_BY_NC_LICENSE_PATH.read_bytes()
    assert (out / ".gitattributes").read_bytes() == publish_hf.GITATTRIBUTES_PATH.read_bytes()
    assert "The audio is not redistributed here" in notice
    assert "CC BY 4.0" in notice and "tarteel-ai/tlog" in notice
    # QUD was an evaluation input, so it is credited in prose but kept out of the training
    # metadata asserted above; the notice names the revision every read was pinned to.
    qud_revision = json.loads(QUD_SOURCES.read_text(encoding="utf-8"))["qud"]["revision"]
    assert "evaluated, not trained" in card
    assert "QUD-Technologies/quranic-universal-ayahs" in card
    assert "QUD Technologies - Qur'anic Universal Audio" in notice
    assert qud_revision in notice


def test_export_refuses_to_attach_the_release_card_to_another_checkpoint(candidate, tmp_path):
    """The card's comparison and readout measured one artifact; a licence is not consent."""
    path, _, _ = candidate
    out = tmp_path / "stage"

    with pytest.raises(SystemExit, match="round-3 candidate"):
        publish_hf.export_and_publish(
            checkpoint_path=path,
            out_dir=out,
            repo_id="sysofwan/hifzguide-muaalem-mini-v2",
            profile="cc-by-nc",
            use_ema=True,
        )
    assert not out.exists() and not out.with_name(out.name + ".tmp").exists()


def test_export_refuses_live_weights_for_the_release_card(
    candidate, release_identity, tmp_path
):
    """The readout scored the averaged weights; exporting the live ones is a different model."""
    path, _, _ = candidate
    out = tmp_path / "stage"

    with pytest.raises(SystemExit, match="weights: got 'live'"):
        publish_hf.export_and_publish(
            checkpoint_path=path,
            out_dir=out,
            repo_id="sysofwan/hifzguide-muaalem-mini-v2",
            profile="cc-by-nc",
            use_ema=False,
        )
    assert not out.exists()


def test_export_pins_the_feature_extractor_to_the_runs_teacher_revision(
    candidate, extractor_calls, tmp_path
):
    path, _, _ = candidate
    publish_hf.export_and_publish(
        checkpoint_path=path, out_dir=tmp_path / "stage", repo_id="x/y", use_ema=True
    )

    assert extractor_calls == [
        {"model_id": publish_hf.TEACHER_MODEL_ID, "revision": TEACHER_REVISION}
    ]

    with pytest.raises(SystemExit, match="not the revision the run trained against"):
        publish_hf.export_and_publish(
            checkpoint_path=path,
            out_dir=tmp_path / "other",
            repo_id="x/y",
            use_ema=True,
            teacher_revision="deadbeef",
        )


def test_export_refuses_a_run_that_records_no_teacher_revision(candidate, tmp_path):
    path, _, _ = candidate
    state = torch.load(path, map_location="cpu", weights_only=False)
    del state["config"]["teacher_revision"]
    unrecorded = tmp_path / "unrecorded.pt"
    torch.save(state, unrecorded)

    with pytest.raises(SystemExit, match="records no teacher_revision"):
        publish_hf.export_and_publish(
            checkpoint_path=unrecorded, out_dir=tmp_path / "stage", repo_id="x/y", use_ema=True
        )


def test_publishing_ema_from_a_live_only_checkpoint_fails_loudly(candidate, tmp_path):
    path, _, _ = candidate
    state = torch.load(path, map_location="cpu", weights_only=False)
    del state["ema_state"]
    live_only = tmp_path / "live_only.pt"
    torch.save(state, live_only)

    with pytest.raises(SystemExit):
        publish_hf.export_and_publish(
            checkpoint_path=live_only, out_dir=tmp_path / "stage", repo_id="x/y", use_ema=True
        )
    assert not (tmp_path / "stage").exists()


def test_export_refuses_a_non_empty_directory(candidate, tmp_path):
    """Writing over an older stage would mix two weight sets under one repository."""
    path, _, _ = candidate
    out = tmp_path / "stage"
    out.mkdir()
    (out / "config.json").write_text("{}", encoding="utf-8")

    with pytest.raises(SystemExit, match="not empty"):
        publish_hf.export_and_publish(
            checkpoint_path=path, out_dir=out, repo_id="x/y", use_ema=True
        )


def test_export_leaves_no_partial_stage_when_staging_fails(candidate, monkeypatch, tmp_path):
    """The rename is the commit: a hub fetch error leaves a `.tmp`, never a half-written stage."""
    path, _, _ = candidate

    def explode(*_args, **_kwargs):
        raise RuntimeError("hub unavailable")

    monkeypatch.setattr(
        publish_hf.SeamlessM4TFeatureExtractor, "from_pretrained", staticmethod(explode)
    )
    out = tmp_path / "stage"
    with pytest.raises(RuntimeError):
        publish_hf.export_and_publish(
            checkpoint_path=path, out_dir=out, repo_id="x/y", use_ema=True
        )
    assert not out.exists()


def test_export_json_record_keeps_stdout_machine_readable(
    candidate, extractor_calls, release_identity, tmp_path, monkeypatch, capsys
):
    path, _, _ = candidate
    out = tmp_path / "stage"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "publish_hf",
            "--checkpoint", str(path),
            "--ema",
            "--out-dir", str(out),
            "--repo-id", "sysofwan/hifzguide-muaalem-mini-v2",
            "--profile", "cc-by-nc",
            "--json",
        ],
    )

    publish_hf.main()

    captured = capsys.readouterr()
    record = json.loads(captured.out)
    assert record["weights"] == "ema" and record["profile"] == "cc-by-nc"
    assert set(record["sha256"]) == EXPECTED_FILES
    assert "Loaded h256" in captured.err


def _release_stage(checkpoint: Path, out: Path) -> Path:
    """A stage of the release card's weights, as the v1.0.7 publication was written."""
    publish_hf.export_and_publish(
        checkpoint_path=checkpoint,
        out_dir=out,
        repo_id="sysofwan/hifzguide-muaalem-mini-v2",
        profile="cc-by-nc",
        use_ema=True,
    )
    return out


def test_restaging_a_card_reproduces_the_stage_byte_for_byte(
    candidate, extractor_calls, release_identity, monkeypatch, tmp_path
):
    """A card correction copies the published files; the re-render must equal the export's.

    The restage reads the parameter count from the weight file's header instead of the loaded
    model, so equality here is the check that the two paths agree and the copy lost nothing.
    """
    stage = _release_stage(candidate[0], tmp_path / "stage")
    monkeypatch.setattr(
        publish_hf,
        "RELEASE_STAGE_SAFETENSORS_SHA256",
        publish_hf.sha256_file(stage / "model.safetensors"),
    )
    out = tmp_path / "restaged"

    record = publish_hf.render_release_card(
        stage, out, "sysofwan/hifzguide-muaalem-mini-v2"
    )

    assert {p.name for p in out.iterdir()} == EXPECTED_FILES
    for name in EXPECTED_FILES:
        assert (out / name).read_bytes() == (stage / name).read_bytes(), name
    assert record["num_params"] == publish_hf.safetensors_param_count(
        stage / "model.safetensors"
    )
    assert record["sha256"] == {
        name: publish_hf.sha256_file(out / name) for name in EXPECTED_FILES
    }


def test_restaging_refuses_a_stage_the_card_did_not_measure(
    candidate, extractor_calls, release_identity, tmp_path
):
    """The card's figures belong to one weight file; another stage would restate them for it."""
    stage = _release_stage(candidate[0], tmp_path / "stage")
    out = tmp_path / "restaged"

    with pytest.raises(SystemExit, match="not the release's"):
        publish_hf.render_release_card(stage, out, "sysofwan/hifzguide-muaalem-mini-v2")
    assert not out.exists() and not out.with_name(out.name + ".tmp").exists()

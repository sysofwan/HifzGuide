"""The published mini's reporting decode (``training.mini_flags``), harness- and GPU-free.

What these tests pin:

* the decode's identity is the card's immutable Hub artifact -- repository, pinned revision and
  ``model.safetensors`` SHA-256 -- and a weights file whose bytes differ is refused before any
  model is loaded;
* a loaded decoder must carry the round's own numerics for every mode (dtype, batch size,
  device, inference policy), or the decode would not be comparable with the round's;
* the identity record written beside the parts names that artifact and every mode's
  fingerprint, so the pooled merge cites the weights that decoded the tranches;
* the job is round 3's two confirm tranches and their gating shards, never another population.
"""

from __future__ import annotations


import pytest

from tadabur.tlog_eval_decode import gating_rows, read_confirm_manifest
from training import mini_flags as mf
from training.aug_readout_spec import CONSONANT3
from training.decoding import DecodeFingerprint
from training.external_decode import MODES
from training.truth_baseline import file_sha256


def _fake_decoder(**overrides):
    """A decoder whose every mode's fingerprint is the expected one, plus ``overrides``."""

    class Fake:
        def fingerprint(self, mode):
            return DecodeFingerprint.from_dict(
                {**mf.expected_fingerprint(mode), **overrides}, "test")

    return Fake()


def test_require_weights_accepts_only_the_pinned_bytes(tmp_path, monkeypatch):
    path = tmp_path / mf.MINI_WEIGHTS
    path.write_bytes(b"not the published mini")
    with pytest.raises(SystemExit, match="not the pinned"):
        mf.require_weights(path)
    monkeypatch.setattr(mf, "MINI_WEIGHTS_SHA256", file_sha256(path))
    mf.require_weights(path)


def test_the_decode_must_carry_the_round_s_numerics():
    mf.require_protocol(_fake_decoder())
    with pytest.raises(SystemExit, match="weights_dtype"):
        mf.require_protocol(_fake_decoder(weights_dtype="fp32"))
    with pytest.raises(SystemExit, match="batch_size"):
        mf.require_protocol(_fake_decoder(batch_size=16))
    with pytest.raises(SystemExit, match="device_type"):
        mf.require_protocol(_fake_decoder(device_type="cpu", autocast=False))
    with pytest.raises(SystemExit, match="policy"):
        mf.require_protocol(_fake_decoder(policy="fp16-features-v0"))


def test_the_identity_record_names_the_pinned_artifact_and_every_mode():
    record = mf.decoder_identity(_fake_decoder())
    assert record["model"] == mf.MINI
    assert record["identity"] == {
        "model_ref": "sysofwan/hifzguide-muaalem-mini",
        "hub_revision": "a2bebf7c2e9a74d9ff5c3269594132d7caea7696",
        "safetensors_sha256": (
            "2570fd3780a8435a11eb20e336bac29ac6f4dedd7c367d39abc20627e6784bb9"),
    }
    assert set(record["fingerprints"]) == set(MODES)
    for arm, fingerprint in record["fingerprints"].items():
        assert fingerprint == mf.expected_fingerprint(MODES[arm])


def test_the_job_is_the_round_s_confirm_tranches_and_their_gating_shards():
    job = mf.plan()
    assert job["roles"] == list(CONSONANT3.confirm_tranches) == ["confirm.3", "confirm.4"]
    expected = {role: len({row["shard_file"]
                           for row in gating_rows(read_confirm_manifest(
                               role, spec=CONSONANT3))})
                for role in CONSONANT3.confirm_tranches}
    assert job["shards"] == expected
    assert set(job["shards"]) == {"confirm.3", "confirm.4"}
    assert all(count > 0 for count in job["shards"].values())


def test_decode_refuses_a_shard_index_outside_its_count(tmp_path):
    with pytest.raises(ValueError, match="shard-index"):
        mf.decode_role("confirm.3", parts_dir=tmp_path, cache_dir=tmp_path,
                       shard_index=4, shard_count=4)

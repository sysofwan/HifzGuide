"""Tests for the full-config parquet-shard reader (:mod:`tadabur.shard_reader`).

These exercise the spec parser and the batched row iterator against a *synthetic* local
parquet shard shaped like Tadabur's ``default`` config (an ``audio`` struct plus the
top-level id ints), so they run offline — no HF download. ``hf_hub_download`` is stubbed
to return the local fixture path.
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from tadabur.shard_reader import (
    NUM_SHARDS,
    iter_shard_rows,
    parse_shard_spec,
)


def _write_shard(path, n_rows: int, base: int = 0) -> None:
    """Write an ``n_rows`` parquet shard with Tadabur's default-config schema."""
    audio = [
        {"bytes": bytes([i % 256]) * 8, "path": f"tadabur_spk0000_S1_A{base + i}_x.wav"}
        for i in range(n_rows)
    ]
    table = pa.table({
        "audio": audio,
        "surah_id": [0] * n_rows,
        "ayah_id": [base + i for i in range(n_rows)],
        "reciter_id": [7] * n_rows,
    })
    pq.write_table(table, path)


# MARK: - parse_shard_spec


def test_parse_shard_spec_single_and_range():
    assert parse_shard_spec("0-4,10,20-21") == [0, 1, 2, 3, 4, 10, 20, 21]


def test_parse_shard_spec_dedupes_and_sorts():
    assert parse_shard_spec("5,3,3,4-5") == [3, 4, 5]


def test_parse_shard_spec_rejects_empty():
    with pytest.raises(ValueError):
        parse_shard_spec("   ")


def test_parse_shard_spec_rejects_reversed_range():
    with pytest.raises(ValueError):
        parse_shard_spec("9-3")


def test_parse_shard_spec_rejects_out_of_range():
    with pytest.raises(ValueError):
        parse_shard_spec(str(NUM_SHARDS))


def test_parse_shard_spec_rejects_non_integer():
    with pytest.raises(ValueError):
        parse_shard_spec("0-x")


# MARK: - iter_shard_rows


def test_iter_shard_rows_yields_all_rows_in_order(tmp_path, monkeypatch):
    shard = tmp_path / "train-00000.parquet"
    _write_shard(shard, 40)
    # iter_shard_rows imports hf_hub_download lazily from huggingface_hub at call time,
    # so patching it there is enough (no network).
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(shard))

    rows = list(iter_shard_rows([0], batch_size=7))
    assert len(rows) == 40  # batching does not drop the final short batch
    assert [r["ayah_id"] for r in rows] == list(range(40))
    assert rows[0]["audio"]["path"] == "tadabur_spk0000_S1_A0_x.wav"
    assert rows[0]["audio"]["bytes"] == bytes([0]) * 8


def test_iter_shard_rows_spans_multiple_shards_in_given_order(tmp_path, monkeypatch):
    s0, s1 = tmp_path / "s0.parquet", tmp_path / "s1.parquet"
    _write_shard(s0, 5, base=0)
    _write_shard(s1, 5, base=100)
    paths = {0: str(s0), 1: str(s1)}
    import huggingface_hub
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download",
        lambda repo, filename, **k: paths[int(filename.split("-")[1].split(".")[0])],
    )

    rows = list(iter_shard_rows([1, 0], batch_size=3))
    # Shard 1 first (as ordered), then shard 0 — reader preserves the given order.
    assert [r["ayah_id"] for r in rows] == [100, 101, 102, 103, 104, 0, 1, 2, 3, 4]


def test_iter_shard_rows_deletes_shard_when_asked(tmp_path, monkeypatch):
    shard = tmp_path / "train-00000.parquet"
    _write_shard(shard, 4)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(shard))

    list(iter_shard_rows([0], batch_size=2, delete_after=True))
    assert not shard.exists()  # blob freed after the shard is consumed

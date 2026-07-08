"""Tests for the per-contrast audit sampler (issue #16): deterministic, seeded,
per-contrast + marginal-band sampling."""

from __future__ import annotations

from tadabur.audit_sampler import (
    MARGINAL_CONTRAST,
    WorklistItem,
    sample_worklist,
    write_worklist,
)
from tadabur.contrast_attribution import all_contrasts
from tadabur.manifest import ManifestRecord, read_records

SHADDA = "shadda"
SOFT = "\u0633\u2194\u0635"  # س↔ص


def _record(name: str, ratio: float, contrasts: tuple[str, ...] = ()) -> ManifestRecord:
    return ManifestRecord(name, "2:1", ratio, 10.0, 7, contrasts=contrasts)


def _shadda_pool(n: int, ratio: float = 0.9) -> list[ManifestRecord]:
    return [_record(f"c{i:03d}.wav", ratio, (SHADDA,)) for i in range(n)]


def test_per_contrast_sample_capped_at_n():
    records = _shadda_pool(100)
    items = sample_worklist(records, per_contrast=30, marginal_n=0, seed=1)
    shadda = [it for it in items if it.contrast == SHADDA]
    assert len(shadda) == 30
    assert all(it.contrast == SHADDA for it in shadda)


def test_takes_all_when_fewer_than_n():
    records = _shadda_pool(5)
    items = sample_worklist(records, per_contrast=30, marginal_n=0, seed=1)
    shadda = [it for it in items if it.contrast == SHADDA]
    assert len(shadda) == 5


def test_sampling_is_deterministic_under_fixed_seed():
    records = _shadda_pool(100)
    a = sample_worklist(records, per_contrast=30, marginal_n=0, seed=42)
    b = sample_worklist(records, per_contrast=30, marginal_n=0, seed=42)
    assert a == b


def test_different_seed_changes_the_draw():
    records = _shadda_pool(100)
    a = {it.clip_id for it in sample_worklist(records, per_contrast=30, marginal_n=0, seed=1)}
    b = {it.clip_id for it in sample_worklist(records, per_contrast=30, marginal_n=0, seed=2)}
    assert a != b


def test_bucket_order_independent():
    # The draw depends only on manifest contents + seed, not on record order.
    records = _shadda_pool(100)
    forward = sample_worklist(records, per_contrast=30, marginal_n=0, seed=7)
    reversed_ = sample_worklist(list(reversed(records)), per_contrast=30, marginal_n=0, seed=7)
    assert forward == reversed_


def test_clip_audited_once_per_contrast_it_exhibits():
    both = _record("multi.wav", 0.9, (SHADDA, SOFT))
    items = sample_worklist([both], per_contrast=30, marginal_n=0, seed=1)
    contrasts = {it.contrast for it in items if it.clip_id == "multi.wav"}
    assert contrasts == {SHADDA, SOFT}


def test_marginal_band_selects_just_above_threshold():
    records = [
        _record("low.wav", 0.60, (SHADDA,)),   # below threshold
        _record("in1.wav", 0.66, (SHADDA,)),   # in band
        _record("in2.wav", 0.71, (SHADDA,)),   # in band
        _record("high.wav", 0.90, (SHADDA,)),  # above band
    ]
    items = sample_worklist(
        records, per_contrast=0, marginal_n=30, threshold=0.65, marginal_band=0.07, seed=1
    )
    marginal = {it.clip_id for it in items if it.contrast == MARGINAL_CONTRAST}
    assert marginal == {"in1.wav", "in2.wav"}


def test_worklist_covers_all_contrast_buckets():
    records = []
    for contrast in all_contrasts():
        records += [_record(f"{contrast}-{i}.wav", 0.9, (contrast,)) for i in range(3)]
    items = sample_worklist(records, per_contrast=30, marginal_n=0, seed=1)
    covered = {it.contrast for it in items}
    assert covered == set(all_contrasts())


def test_worklist_item_shape():
    items = sample_worklist(_shadda_pool(1), per_contrast=1, marginal_n=0, seed=1)
    (item,) = [it for it in items if it.contrast == SHADDA]
    assert isinstance(item, WorklistItem)
    assert item.clip_id == "c000.wav"
    assert item.audio_ref == "c000.wav"
    assert item.match_ratio == 0.9


def test_write_and_reload_worklist_round_trip(tmp_path):
    records = _shadda_pool(3)
    items = sample_worklist(records, per_contrast=30, marginal_n=0, seed=1)
    path = tmp_path / "worklist.jsonl"
    write_worklist(items, path)
    lines = [line for line in path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == len(items)


def test_sampler_reads_manifest_records(tmp_path):
    manifest = tmp_path / "subset.jsonl"
    manifest.write_text(
        '{"audio_filename": "x.wav", "surah_ayah": "2:1", "match_ratio": 0.9, '
        '"ayah_duration_s": 10.0, "reciter_id": 7, "contrasts": ["shadda"]}\n',
        encoding="utf-8",
    )
    items = sample_worklist(read_records(manifest), per_contrast=30, marginal_n=0, seed=1)
    assert [it.clip_id for it in items if it.contrast == SHADDA] == ["x.wav"]

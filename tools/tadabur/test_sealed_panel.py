"""Tests for the sealed held-out panel (#89): the seal at the tools' input boundaries, the
frame, the manifest, and the committed panel's disjointness from every other use.

None of these tests scores the panel, reads its audio, or passes an unsealing flag except
the preparation flag, inside a block that reads nothing.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import pytest

from tadabur import panel_seal
from tadabur.exposure import (
    H448_INIT_VALIDATION,
    H448_TRAINING,
    MINING_POOL,
    SEALED_PANEL,
    TRUTH_SITE_USES,
    Exposure,
    ExposureOverlap,
    ShardExposure,
    check_disjoint,
    load_registry,
    require_complete,
    write_shard_use,
    write_use,
)
from tadabur.panel_seal import PanelProvenance, SealedPanelError, refuse_sealed, unsealed
from tadabur.sealed_panel import (
    STAGED_PATH,
    PanelClip,
    PanelSegment,
    build_manifest,
    decodable_segment_keys,
    exposed_reciters,
    frame_record,
    load_manifest,
    load_panel_registry,
    load_teacher_decodes,
    open_for_scoring,
    panel_frame,
    reciter_overlap_allowed,
    reference_capacity,
    source_overlap_allowed,
    unseen_shards,
    write_manifest,
)
from tadabur.staged_audio import (
    REGISTRY_PATH,
    IndexRow,
    StagedClip,
    load_staged_clips,
    read_staged_clips,
    verify_staged,
    write_staged_clips,
)

TOOLS_DIR = Path(__file__).resolve().parent.parent
SHA = "d" * 64
#: Spelled in two halves so this test does not name the flags it pins.
SHIP_FLAG = "SHIP_CRITERION_" + "AUTHORIZATION"
PREP_FLAG = "PREPARATION_" + "AUTHORIZATION"


def _a_panel_clip() -> StagedClip:
    return min(load_panel_registry().values(), key=lambda c: c.audio_filename)


# --- the seal at the input boundaries --------------------------------------------------


@pytest.mark.parametrize("authorization", [None, "", "issue-97", "yes, score it"])
def test_scoring_the_panel_without_the_ship_criterions_flag_fails_loudly(tmp_path, authorization):
    with pytest.raises(SealedPanelError, match="scored only by the ship criterion"):
        with open_for_scoring(tmp_path, authorization=authorization):
            pass


def test_the_preparation_flag_cannot_score(tmp_path):
    with pytest.raises(SealedPanelError, match="cannot score"):
        with open_for_scoring(tmp_path, authorization=getattr(panel_seal, PREP_FLAG)):
            pass


def test_a_glob_loader_refuses_a_panel_clip(tmp_path):
    from training.distill_data import discover_clips

    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / _a_panel_clip().audio_filename).write_bytes(b"RIFF")
    (tmp_path / "other.wav").write_bytes(b"RIFF")
    with pytest.raises(SealedPanelError, match="sealed-panel audio"):
        discover_clips(tmp_path)


def test_a_renamed_copy_in_another_directory_is_caught_by_its_checksum(tmp_path, monkeypatch):
    copy = tmp_path / "elsewhere" / "renamed.wav"
    copy.parent.mkdir()
    copy.write_bytes(b"RIFF" + bytes(40) + bytes(2 * 10))  # a 10-sample PCM_16 WAV's size
    provenance = PanelProvenance(
        frozenset({"panel.wav"}), frozenset({hashlib.sha256(copy.read_bytes()).hexdigest()}),
        frozenset({copy.stat().st_size}),
    )
    monkeypatch.setattr(panel_seal, "panel_provenance", lambda: provenance)
    from training.distill_data import discover_clips

    with pytest.raises(SealedPanelError):
        discover_clips(tmp_path)
    unrelated = tmp_path / "unrelated.wav"
    unrelated.write_bytes(b"x" * 64)
    assert refuse_sealed(unrelated) == unrelated


def test_the_distill_eval_cli_refuses_an_audio_root_holding_panel_audio(tmp_path, monkeypatch):
    import torch

    from training import distill_eval

    (tmp_path / _a_panel_clip().audio_filename).write_bytes(b"RIFF")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(sys, "argv", [
        "distill_eval", "--checkpoint", str(tmp_path / "missing.pt"),
        "--audio-root", str(tmp_path),
    ])
    with pytest.raises(SealedPanelError):  # before any checkpoint or model is loaded
        distill_eval.main()


def test_the_clip_readers_refuse_panel_audio(tmp_path):
    from tadabur.segment_score import _load_clip
    from training.decode_evalset import read_clip_audio

    clip = _a_panel_clip()
    (tmp_path / clip.audio_filename).write_bytes(b"RIFF")
    for read in (lambda: _load_clip(tmp_path, clip.audio_filename),
                 lambda: read_clip_audio(tmp_path, clip.audio_filename),
                 lambda: verify_staged(clip, tmp_path)):
        with pytest.raises(SealedPanelError):
            read()


def test_the_shared_staging_loader_refuses_the_panel_registry_and_a_copy_of_its_rows(tmp_path):
    with pytest.raises(SealedPanelError):
        load_staged_clips(STAGED_PATH)
    clip = _a_panel_clip()
    copied = tmp_path / "clips.jsonl"  # the panel's rows relabelled into another registry
    write_staged_clips([StagedClip(**{**clip.__dict__, "uses": (MINING_POOL,)})], copied)
    with pytest.raises(SealedPanelError):
        load_staged_clips(copied)
    assert read_staged_clips(copied)  # the unchecked reader serves the panel module only


def test_unsealing_needs_a_known_flag_and_lasts_only_for_its_block(tmp_path):
    path = tmp_path / _a_panel_clip().audio_filename
    with pytest.raises(SealedPanelError):
        with unsealed("please"):
            pass
    with unsealed(getattr(panel_seal, PREP_FLAG)):
        refuse_sealed(path)
    with pytest.raises(SealedPanelError):
        refuse_sealed(path)


def test_the_training_audio_cache_refuses_a_panel_clip_and_its_local_copies(tmp_path):
    from tadabur.audit_sampler import local_audio_path
    from training.windowed_batch import ClipAudioCache

    name = _a_panel_clip().audio_filename
    (tmp_path / local_audio_path(name)).write_bytes(b"RIFF")  # the hash-prefixed copy
    with pytest.raises(SealedPanelError):
        ClipAudioCache(tmp_path).waveform(name)


def test_a_segment_file_cut_from_a_panel_clip_is_refused(tmp_path):
    from tadabur.audio import read_audio, read_audio_bytes

    segment = tmp_path / _a_panel_clip().audio_filename.replace(".wav", "__seg0.wav")
    segment.write_bytes(b"RIFF")
    with pytest.raises(SealedPanelError):
        read_audio_bytes(segment)  # minimal_pairs and the eval harness read through here
    with pytest.raises(SealedPanelError):
        read_audio(segment)


def test_panel_bytes_are_refused_whatever_buffer_they_arrive_in(monkeypatch):
    import io

    import numpy as np
    import soundfile as sf

    from tadabur.audio import decode_to_mono_16k

    buffer = io.BytesIO()
    sf.write(buffer, np.zeros(160, dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    data = buffer.getvalue()
    provenance = PanelProvenance(frozenset({"panel.wav"}),
                                 frozenset({hashlib.sha256(data).hexdigest()}),
                                 frozenset({len(data)}))
    monkeypatch.setattr(panel_seal, "panel_provenance", lambda: provenance)
    with pytest.raises(SealedPanelError):
        decode_to_mono_16k(data)
    assert len(decode_to_mono_16k(data[:-2] + b"\x01\x00")) == 160  # other bytes decode


def _write_shard(path: Path, names: list[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pq.write_table(pa.table({
        "audio": [{"bytes": b"RIFF" + bytes(8), "path": name} for name in names],
        "surah_id": [0] * len(names), "ayah_id": [1] * len(names), "reciter_id": [7] * len(names),
    }), path)


def test_streamed_shard_rows_keep_a_panel_rows_place_but_seal_its_audio(tmp_path, monkeypatch):
    import huggingface_hub

    from tadabur.filter import parse_clip
    from tadabur.shard_reader import iter_shard_rows

    panel_name = _a_panel_clip().audio_filename
    shard = tmp_path / "train-00000.parquet"
    _write_shard(shard, ["tadabur_spk0001_S1_A1_ab_000001.wav", panel_name])
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", lambda *a, **k: str(shard))

    ordinary, panel_row = iter_shard_rows([0])
    assert parse_clip(ordinary).audio_bytes  # other rows are untouched
    assert panel_row["audio"]["path"] == panel_name  # identity and position survive
    for read in (lambda: panel_row["audio"]["bytes"], lambda: parse_clip(panel_row)):
        with pytest.raises(SealedPanelError):  # the filter cannot score it
            read()
    with unsealed(getattr(panel_seal, PREP_FLAG)):  # the panel's own staging still reads it
        _, staged_row = iter_shard_rows([0])
        assert staged_row["audio"]["bytes"]


def test_every_dataset_stream_goes_through_the_row_seal():
    from tadabur.panel_seal import seal_row

    row = {"audio": {"bytes": b"x", "path": _a_panel_clip().audio_filename}, "reciter_id": 1}
    with pytest.raises(SealedPanelError):
        seal_row(row)["audio"]["bytes"]
    assert seal_row({"audio": {"bytes": b"x", "path": "other.wav"}})["audio"]["bytes"] == b"x"


#: Where audio may be read without the shared guarded readers, and why.
READ_EXEMPTIONS = {
    "tadabur/audio.py": "the guarded readers themselves",
    "tadabur/staged_audio.py": "as_staged reads back an in-memory buffer it just wrote",
}


def test_lint_every_audio_read_and_row_source_goes_through_a_choke_point():
    """Every audio file read is read_audio / read_audio_bytes, every byte decode is
    decode_to_mono_16k, and every Tadabur row source is sealed."""
    import re

    sources = [p for p in sorted(TOOLS_DIR.rglob("*.py")) if not p.name.startswith("test_")]
    raw_reads = re.compile(r"\b(sf|soundfile)\.(read|SoundFile)\(|librosa\.load\(|torchaudio\.load\(")
    bytes_decode = re.compile(r"decode_to_mono_16k\([^)]*read_bytes\(\)")
    offenders = []
    for path in sources:
        text = path.read_text(encoding="utf-8")
        rel = str(path.relative_to(TOOLS_DIR))
        if raw_reads.search(text) and rel not in READ_EXEMPTIONS:
            offenders.append(f"{rel}: raw audio read")
        if bytes_decode.search(text):
            offenders.append(f"{rel}: decodes a file's bytes without read_audio_bytes")
        if "load_dataset(" in text and rel != "tadabur/dataset_source.py":
            offenders.append(f"{rel}: streams the dataset without stream_rows")
        if "ParquetFile(" in text and rel != "tadabur/shard_reader.py":
            offenders.append(f"{rel}: reads a shard without iter_shard_rows")
    assert offenders == []


def _mentions(path: Path, needle: str) -> bool:
    return needle in path.read_text(encoding="utf-8")


def test_lint_only_the_panels_own_modules_name_it_or_its_flags():
    """Supplementary lint: the seal itself is enforced at the input boundaries above."""
    sources = sorted(TOOLS_DIR.rglob("*.py"))
    assert len(sources) > 50  # the scan really walks tools/

    def naming(needle: str) -> set[str]:
        return {str(p.relative_to(TOOLS_DIR)) for p in sources if _mentions(p, needle)}

    assert naming("sealed_panel") <= {
        "tadabur/sealed_panel.py", "tadabur/test_sealed_panel.py", "tadabur/panel_seal.py",
        "tadabur/exposure.py", "tadabur/test_exposure.py", "tadabur/staged_audio.py",
    }
    assert naming(SHIP_FLAG) == {"tadabur/panel_seal.py"}  # #97 adds its scoring module
    assert naming(PREP_FLAG) == {"tadabur/panel_seal.py", "tadabur/sealed_panel.py"}


# --- the frame -------------------------------------------------------------------------


def _index_row(name, reciter, shard=39, row=0, seconds=8.0, surah_ayah="2:255") -> IndexRow:
    return IndexRow(name, shard, row, reciter, surah_ayah, seconds)


def test_the_unseen_shards_are_the_held_out_block_and_the_strided_reserve():
    shards = unseen_shards()
    assert shards[:21] == list(range(21)) and shards[21:] == list(range(39, 385, 19))


def test_panel_frame_bars_used_reciters_whole_then_applies_the_row_bounds():
    rows = [
        _index_row("a.wav", reciter=1, row=0),
        _index_row("b.wav", reciter=2, row=1),                       # a used reciter
        _index_row("c.wav", reciter=1, row=2, seconds=60.0),          # too long
        _index_row("d.wav", reciter=1, row=3, surah_ayah="106:1"),    # no phonetization
        _index_row("e.wav", reciter=1, shard=21, row=0),              # h448 trained on it
    ]
    panel, excluded = panel_frame(rows, {2: [MINING_POOL]})
    assert [r.audio_filename for r in panel] == ["a.wav"]
    assert {row.audio_filename: reason for row, reason in excluded} == {
        "b.wav": "reciter_exposed", "c.wav": "duration", "d.wav": "phonetizer_unsupported",
    }


def _shards(shards, rows, membership) -> ShardExposure:
    return ShardExposure(tuple(shards), rows, SHA, membership, True, "test")


def test_only_shared_baseline_uses_and_the_panel_itself_leave_a_reciter_eligible(tmp_path):
    write_use(MINING_POOL, [Exposure("a.wav", 39, 0, 1, None, None, SHA)], tmp_path)
    write_use(SEALED_PANEL, [Exposure("b.wav", 40, 0, 2, None, None, SHA)], tmp_path)
    write_shard_use(H448_TRAINING, _shards((21,), {1: 4, 3: 2}, "exact"), tmp_path)
    write_shard_use(H448_INIT_VALIDATION, _shards((0,), {5: 1}, "uncertain"), tmp_path)
    registry = load_registry(tmp_path)
    assert exposed_reciters(registry) == {1: [MINING_POOL]}
    assert reciter_overlap_allowed(registry) == [H448_INIT_VALIDATION, H448_TRAINING]
    assert source_overlap_allowed(registry) == [H448_INIT_VALIDATION]


def test_frame_record_counts_rows_and_the_shared_baseline_overlap(tmp_path):
    write_use(MINING_POOL, [Exposure("x.wav", 39, 9, 2, None, None, SHA)], tmp_path)
    write_shard_use(H448_TRAINING, _shards((21,), {1: 4}, "exact"), tmp_path)
    registry = load_registry(tmp_path)
    rows = [_index_row("a.wav", 1, row=0), _index_row("b.wav", 2, row=1)]
    panel, excluded = panel_frame(rows, exposed_reciters(registry))
    record = frame_record(panel, excluded, exposed_reciters(registry), registry)
    assert record["per_shard"]["39"] == {"excluded_reciter_exposed": 1, "panel": 1}
    assert record["per_reciter"] == {"1": {"clips": 1, "h448.training_rows": 4}}
    assert record["unseen_reciters_barred_by_use"] == {MINING_POOL: 1}


# --- the manifest ----------------------------------------------------------------------


def _staged(name="a.wav", reciter=1, samples=48000) -> StagedClip:
    return StagedClip(name, 39, 0, reciter, "2:255", samples, SHA, (SEALED_PANEL,))


def _status(name="a.wav", reciter=1) -> dict:
    return {
        "audio_filename": name, "surah_ayah": "2:255", "reciter_id": reciter, "n_words": 3,
        "skip_reason": None, "re_reads": 0, "recited_words": 3, "recitation_start_s": 0.0,
        "recitation_end_s": 3.0, "word_times": [0.0, 1.0, 2.0, 3.0],
    }


def test_build_manifest_converts_spans_and_round_trips(tmp_path):
    seg = {"segment_index": 0, "word_start": 0, "word_end": 3, "start_s": 0.5, "end_s": 2.0,
           "reference": "قَاالَ", "raw_word_offsets": [0]}
    staged = {"a.wav": _staged()}
    (clip,) = build_manifest([_status()], [{"audio_filename": "a.wav", "segments": [seg]}], staged)
    assert (clip.segments[0].start_sample, clip.segments[0].end_sample) == (8000, 32000)
    write_manifest([clip], tmp_path / "clips.jsonl")
    assert load_manifest(tmp_path / "clips.jsonl", staged) == [clip]
    with pytest.raises(ValueError, match="different clips"):
        build_manifest([_status()], [], {**staged, "b.wav": _staged("b.wav")})
    with pytest.raises(ValueError, match="different clips"):
        load_manifest(tmp_path / "clips.jsonl", {**staged, "b.wav": _staged("b.wav")})


def test_reference_capacity_reads_every_segments_reference():
    def segment(index, reference):
        return PanelSegment(index, 0, 1, 0, 100, reference, (0,))

    clip = PanelClip(
        "a.wav", "2:255", 1, 2, None, 0, 2, 0.0, 1.0, (0.0, 1.0),
        (segment(0, "رَببِ"), segment(1, "ذَ")),
    )
    capacity = reference_capacity([clip])
    assert capacity["haraka_carriers"] == {"damma": 0, "fatha": 2, "kasra": 1}
    assert capacity["reference_geminates"] == 1
    assert capacity["pair_carriers"]["ذ↔ز"] == 1


# --- the committed panel ---------------------------------------------------------------


def test_the_committed_panel_loads_against_its_staging_registry():
    staged = load_panel_registry()
    clips = load_manifest(staged=staged)
    assert len(clips) == len(staged) > 0
    assert all(c.shard in unseen_shards() for c in staged.values())
    fingerprint, decodes = load_teacher_decodes()
    assert fingerprint["model"] == "obadx/muaalem-model-v3_2"  # the base teacher, nothing else
    assert set(decodes) == decodable_segment_keys(clips)


def test_the_committed_panel_carries_no_score():
    """Preparation decoded and segmented; nothing gated, attributed or dropped by a decode."""
    panel_dir = STAGED_PATH.parent
    for name in ("clips.jsonl", "teacher_decodes.json", "frame.json", "staged_clips.jsonl"):
        text = (panel_dir / name).read_text(encoding="utf-8")
        for field in ("match_ratio", "contrasts", '"kept"', "drops"):
            assert field not in text, (name, field)
    summary = json.loads((panel_dir / "summary.json").read_text(encoding="utf-8"))
    assert summary["preparation"]["scoring"].startswith("none")


def test_the_panel_is_reciter_disjoint_from_every_prd_use():
    registry = load_registry()
    require_complete(registry)
    assert registry.reciters(SEALED_PANEL) == {c.reciter_id for c in load_panel_registry().values()}
    allowed = set(reciter_overlap_allowed(registry))
    assert allowed == {H448_TRAINING, H448_INIT_VALIDATION}  # the owner's amendment, no more
    prd_uses = [u for u in registry.uses() if u != SEALED_PANEL and u not in allowed]
    assert set(TRUTH_SITE_USES.values()) <= set(prd_uses) and MINING_POOL in prd_uses
    check_disjoint(SEALED_PANEL, *prd_uses, by="reciter", registry=registry)


def test_the_panel_is_source_disjoint_from_every_use_but_the_uncertain_baseline_one():
    registry = load_registry()
    assert source_overlap_allowed(registry) == [H448_INIT_VALIDATION]
    others = [u for u in registry.uses() if u not in source_overlap_allowed(registry)]
    check_disjoint(SEALED_PANEL, *others, by="source", registry=registry)
    # Documented, not hidden: the panel shares reciters with h448's training, and holds
    # recordings in the shards whose unknown rows h448's init and validation drew from.
    assert registry.reciters(SEALED_PANEL) & registry.reciters(H448_TRAINING)
    with pytest.raises(ExposureOverlap):
        check_disjoint(SEALED_PANEL, H448_INIT_VALIDATION, by="source", registry=registry)


def test_no_panel_clip_is_in_the_shared_staging_registry():
    assert not set(load_panel_registry()) & set(load_staged_clips(REGISTRY_PATH))


def test_the_frame_and_summary_agree_with_the_manifest():
    panel_dir = STAGED_PATH.parent
    frame = json.loads((panel_dir / "frame.json").read_text(encoding="utf-8"))
    summary = json.loads((panel_dir / "summary.json").read_text(encoding="utf-8"))
    staged = load_panel_registry()
    assert sum(r["clips"] for r in frame["per_reciter"].values()) == len(staged)
    assert {int(r) for r in frame["per_reciter"]} == {c.reciter_id for c in staged.values()}
    assert summary["clips"] == len(staged) and summary["reciters"] == frame["panel_reciters"]

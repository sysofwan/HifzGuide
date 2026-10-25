"""Build-path tests for tlog S4 (``tadabur.tlog_overlap.build``, #189).

The external pass cannot run offline — it streams real evaluation audio — so these tests
fabricate its store exactly as :func:`tadabur.tlog_overlap.external` writes it (one npz per
unit, the control candidates, the frame sets' revision stamp) and drive ``build`` end to end
over one real committed index shard and its real split shard. They pin the three defects the
self-check found in that path: the census that expected the frame stores to be empty, the eval
index that opened the wrong store filename, and the frame sets' unpinned revision.

Test command, from ``tools/``::

    python -m pytest -q tadabur/test_tlog_overlap_build.py
"""

from __future__ import annotations

import hashlib
import json
import shutil
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

import tadabur.tlog_overlap as to
from tadabur.audio import DecodeError
from tadabur.nonpro_sources import QDAT, QDAT_BENCH, RETASY, TARTEEL_V1
from tadabur.panel_seal import PanelProvenance
from tadabur.qud_source import audio_filename as qud_audio_filename
from tadabur.tlog_index import INDEX_DIR, write_shard
from tadabur.tlog_pass1 import FLOORS_PATH, read_jsonl_gz
from tadabur.tlog_split import SPLIT_DIR

SPLIT = "unclean"
SHARD = "unclean-00000-of-00332"
ROWS = "unclean-00000-of-00332.jsonl.gz"
#: The frame pass's pinned commit, and another one for the resume-mismatch case.
REVISION = "a" * 40
OTHER_REVISION = "b" * 40
#: Clips per fabricated set: exactly the control's copy count, so every control is available.
MEMBERS = to.CONTROL_COPIES
NONPRO_SETS = {"tarteel_v1": TARTEEL_V1.repo, "qdat": QDAT.repo,
               "qdat_bench": QDAT_BENCH.repo, "retasy": RETASY.repo}
EMPTY_PANEL = PanelProvenance(frozenset(), frozenset(), frozenset())


def _signal(seed: int, seconds: float = 1.2) -> np.ndarray:
    """A seeded burst of low-passed noise: long enough for more than ``MIN_VOTES`` frames.

    A fixed tone stack would be wrong here: two such stacks at one ``f0`` fingerprint only
    ~0.15 apart, under the line, so every pair of fabricated clips would "match". Distinct
    noise seeds sit at ~0.5, like two different recordings.
    """
    rng = np.random.default_rng(seed)
    audio = np.convolve(rng.standard_normal(round(seconds * 16000)), np.ones(32) / 32,
                        mode="same")
    return (audio / np.abs(audio).max() * 0.7).astype(np.float32)


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _ext_sha(code: str, clip: str) -> str:
    return hashlib.sha256(f"{code}:{clip}".encode()).hexdigest()


def _save(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
                    encoding="utf-8")


@dataclass
class Offline:
    """The fabricated inputs ``build`` reads, all inside one test's ``tmp_path``."""

    fp: Path
    store: Path
    index: Path
    pass1: Path
    split: Path
    out: Path
    census: Path
    units: dict[str, Path]
    members: dict[str, list[str]]

    def build(self, **overrides) -> dict:
        return to.build(self.out, fp_dir=self.fp, store_dir=self.store,
                        index_dir=self.index, pass1_dir=self.pass1,
                        floors_path=FLOORS_PATH, split_dir=self.split,
                        code_sha="0" * 40, pull_request=7, **overrides)

    def rewrite_unit(self, code: str, prints: dict[str, np.ndarray]) -> None:
        _save(self.units[code], prints)

    def rewrite_census(self) -> None:
        """The committed ``fingerprints.json`` census of the fabricated tlog store."""
        path = self.store / f"{SHARD}.npz"
        with np.load(path) as data:
            members = len(data.files)
        self.census.write_text(json.dumps({
            "schema_version": 1, "store": str(self.store),
            "npz_key": "audio_path|audio_sha256",
            "shards": {f"{SPLIT}/{SHARD}": {"sha256": _sha256(path), "members": members,
                                           "codec_ok_rows": members}}}), encoding="utf-8")


@pytest.fixture
def offline(tmp_path, monkeypatch):
    """One real index shard plus a whole fabricated external store and tlog fingerprint store."""
    fp, store, index, pass1, split, out = (tmp_path / name for name in
                                           ("fp", "store", "index", "pass1", "split", "out"))
    (index / SPLIT).mkdir(parents=True)
    (split / SPLIT).mkdir(parents=True)
    shutil.copy(INDEX_DIR / SPLIT / ROWS, index / SPLIT / ROWS)
    shutil.copy(SPLIT_DIR / SPLIT / ROWS, split / SPLIT / ROWS)
    index_rows = read_jsonl_gz(index / SPLIT / ROWS)
    split_rows = read_jsonl_gz(split / SPLIT / ROWS)
    roles = [row["role"] for row in split_rows]
    trains = [position for position, role in enumerate(roles) if role == "train"]
    look = roles.index("look")
    exact = (trains[0], trains[1])
    near = (trains[2], look)
    fingerprint_hit, checksum_hit = trains[3], trains[4]

    # Every external set: one unit of MEMBERS indexed clips with their control waveforms.
    members = {code: [f"{code}-{position:03d}.wav" for position in range(MEMBERS)]
               for code in NONPRO_SETS}
    members["qud"] = [qud_audio_filename("pilot", 1, ayah) for ayah in range(1, MEMBERS + 1)]
    members["gate_eval"] = [f"{position}:0" for position in range(MEMBERS)]
    members["tadabur_frame"] = [f"{100 + position}:0" for position in range(MEMBERS)]
    units: dict[str, Path] = {}
    waves: dict[str, dict[str, np.ndarray]] = {}
    for order, code in enumerate(to.SETS):
        waves[code] = {clip: _signal(5_000 + 100 * order + position)
                       for position, clip in enumerate(members[code])}
        _save(fp / code / f"{code}.npz",
              {clip: to.fingerprint(wave) for clip, wave in waves[code].items()})
        _save(fp / code / "control" / f"{code}.npz", waves[code])
        units[code] = fp / code / f"{code}.npz"
    for code in to.FRAME_SETS:
        (fp / code / to.REVISION_NAME).write_text(REVISION + "\n", encoding="utf-8")

    # Committed checksum lists: the external store's population, with distinct checksums.
    nonpro = tmp_path / "nonpro.jsonl"
    _jsonl(nonpro, [{"dataset": repo, "audio_filename": clip,
                     "audio_sha256": _ext_sha(code, clip)}
                    for code, repo in NONPRO_SETS.items() for clip in members[code]])
    qud = tmp_path / "qud.jsonl"
    _jsonl(qud, [{"slug": "pilot", "surah_ayah": f"1:{ayah}",
                  "mp3_sha256": _ext_sha("qud", members["qud"][ayah - 1])}
                 for ayah in range(1, MEMBERS + 1)])
    gate_paths = (tmp_path / "dev.jsonl", tmp_path / "test.jsonl")
    for half, path in enumerate(gate_paths):
        _jsonl(path, [{"shard": MEMBERS * half + position, "row_index": 0,
                       "audio_filename": f"g{MEMBERS * half + position}.wav"}
                      for position in range(MEMBERS // 2)])

    # The tlog side: pass-1 facts aligned with the index shard, and S3b's fingerprint npz.
    shas = {position: hashlib.sha256(f"row:{position}".encode()).hexdigest()
            for position in range(len(index_rows))}
    shas[exact[1]] = shas[exact[0]]
    shas[checksum_hit] = _ext_sha("retasy", members["retasy"][0])
    facts = [{"schema_version": 1, "split": SPLIT, "shard_file": SHARD, "row": position,
              "audio_path": row["audio_path"], "audio_sha256": shas[position],
              "codec_ok": True, "speech_s": 3.0, "speech_ratio": 0.9}
             for position, row in enumerate(index_rows)]
    write_shard(facts, pass1 / SPLIT / ROWS)
    (pass1 / "meta.json").write_text("{}\n", encoding="utf-8")

    prints = {position: to.fingerprint(_signal(20_000 + position))
              for position in range(len(index_rows))}
    prints[near[0]] = prints[near[1]] = to.fingerprint(_signal(9_000))
    prints[exact[0]] = prints[exact[1]] = to.fingerprint(_signal(9_001))
    prints[fingerprint_hit] = to.fingerprint(waves["tarteel_v1"][members["tarteel_v1"][0]])
    store.mkdir(parents=True)
    _save(store / f"{SHARD}.npz",
          {f"{facts[position]['audio_path']}|{shas[position]}": prints[position]
           for position in range(len(facts))})

    census = tmp_path / "fingerprints.json"
    monkeypatch.setattr(to, "NONPRO_CHECKSUMS_PATH", nonpro)
    monkeypatch.setattr(to, "QUD_CHECKSUMS_PATH", qud)
    monkeypatch.setattr(to, "GATE_EVAL_PATHS", gate_paths)
    monkeypatch.setattr(to, "FP_RECORD_PATH", census)
    # Control A re-encodes through MP3; offline the copy is the clip, so it must be found.
    monkeypatch.setattr("tadabur.qud_overlap.synthetic_copy",
                        lambda samples, cut_ms: samples)
    fabricated = Offline(fp, store, index, pass1, split, out, census, units, members)
    fabricated.rewrite_census()
    return fabricated


def test_the_build_writes_every_row_from_both_stores(offline):
    meta = offline.build()

    records = read_jsonl_gz(offline.out / SPLIT / ROWS)
    assert meta["counts"]["rows"] == len(records) == len(read_jsonl_gz(offline.index / SPLIT / ROWS))
    assert [record["row"] for record in records] == list(range(len(records)))
    assert all(record["schema_version"] == to.SCHEMA_VERSION for record in records)

    tarteel = offline.members["tarteel_v1"][0]
    assert meta["hits"]["by_set"] == {"retasy": 1, "tarteel_v1": 1}
    hits = {record["eval_overlap"][0]["set"]: record for record in records
            if record["eval_overlap"]}
    assert hits["tarteel_v1"]["eval_overlap"] == [
        {"set": "tarteel_v1", "clip": tarteel, "ber": 0.0}]
    assert hits["tarteel_v1"]["excluded"] == "eval_overlap:tarteel_v1"
    assert hits["retasy"]["excluded"] == "eval_overlap:retasy"
    assert hits["retasy"]["eval_overlap"][0]["clip"] == offline.members["retasy"][0]

    assert meta["duplicates"] == {"exact": {"train:train": 1}, "near": {"look:train": 1}}
    loser = next(record for record in records
                 if record["duplicate_ber"] not in (None, 0) and record["duplicate_of"])
    assert loser["duplicate_ber"] == 1e-4 and loser["excluded"] == "duplicate"

    for code in to.FRAME_SETS:
        assert meta["sets"][code]["revision"] == REVISION
        assert meta["sets"][code]["checksums"] is None
    assert meta["sets"]["tarteel_v1"]["revision"] == TARTEEL_V1.revision

    recall = meta["controls"]["synthetic_recall"]
    assert recall["copies"] == MEMBERS * len(to.SETS)
    assert recall["found"] == recall["copies"] and recall["max_ber"] == 0.0
    histogram = meta["ber_histogram"]
    assert len(histogram["rejected"]) == len(histogram["counts"]) == len(histogram["bins"])
    assert meta["blind_spots"]["unfingerprinted"] == {}
    assert meta["tlog_store"]["census"]["sha256"] == _sha256(offline.census)
    assert meta["frozen"]["pass1_meta_sha256"] == _sha256(offline.pass1 / "meta.json")

    for counts in meta["net_sizes"]["usable_before_s4"].values():
        assert counts["lost"] == 0  # no eval-role row loses to a training copy
        assert counts["remaining"] == counts["before"]
    assert any(counts["before"] for counts in meta["net_sizes"]["usable_before_s4"].values())


def test_the_build_records_a_committed_clip_without_a_fingerprint(offline):
    code, dropped = "tarteel_v1", offline.members["tarteel_v1"][3]
    with np.load(offline.units[code]) as data:
        prints = {name: data[name] for name in data.files if name != dropped}
    offline.rewrite_unit(code, prints)

    meta = offline.build()

    assert meta["blind_spots"]["unfingerprinted"] == {code: [dropped]}
    assert meta["sets"][code]["members"] == MEMBERS - 1


def test_the_build_refuses_a_store_clip_the_checksums_do_not_name(offline):
    with np.load(offline.units["qdat"]) as data:
        prints = {name: data[name] for name in data.files}
    prints["qdat-intruder.wav"] = to.fingerprint(_signal(1))

    offline.rewrite_unit("qdat", prints)

    with pytest.raises(ValueError, match="qdat-intruder.wav"):
        offline.build()


def test_the_build_refuses_a_frame_store_at_another_revision(offline):
    (offline.fp / "tadabur_frame" / to.REVISION_NAME).write_text(
        OTHER_REVISION + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="never mix revisions"):
        offline.build()


def test_the_build_refuses_an_unstamped_frame_store(offline):
    (offline.fp / "gate_eval" / to.REVISION_NAME).unlink()

    with pytest.raises(ValueError, match=to.REVISION_NAME):
        offline.build()


def test_the_build_refuses_a_tlog_store_that_is_not_the_census(offline):
    with np.load(offline.store / f"{SHARD}.npz") as data:
        prints = {name: data[name] for name in data.files}
    prints["audio/x|" + "0" * 64] = to.fingerprint(_signal(1))
    _save(offline.store / f"{SHARD}.npz", prints)

    with pytest.raises(ValueError, match="committed census"):
        offline.build()


def test_the_frame_pass_refuses_a_store_of_another_or_unknown_revision(offline):
    with pytest.raises(ValueError, match="fingerprinted at"):
        to._pin_frame_revision(offline.fp, OTHER_REVISION)

    (offline.fp / "gate_eval" / to.REVISION_NAME).unlink()
    with pytest.raises(ValueError, match="cannot be verified"):
        to._pin_frame_revision(offline.fp, REVISION)


def test_verify_reports_the_candidates_the_line_rejects():
    real = to.fingerprint(_signal(21))
    index = to.EvalIndex.build({"tarteel_v1:real.wav": real})
    probe = real.copy()
    for position in range(len(probe)):
        if position % 5:  # one word in five stays intact: still MIN_VOTES frames of votes
            probe[position] ^= 0x00FF00FF

    hits = index.verify(probe)

    assert [hit.clip for hit in hits] == ["tarteel_v1:real.wav"]
    assert hits[0].ber > to.LINE
    assert index.query(probe) == []
    stats = {"verified": 0, "mostly_stop_word": 0, "histogram": Counter(),
             "rejected": Counter()}
    assert to._count_hit(stats, hits[0]) is False
    assert stats["verified"] == 1 and stats["histogram"].total() == 0
    assert stats["rejected"].total() == 1


def test_a_unit_offers_no_control_candidate_the_index_cannot_hold(tmp_path):
    silent = np.zeros(2 * 16000, dtype=np.float32)
    unit = to._Unit("unit", iter([("loud.wav", _signal(11)), ("silent.wav", silent)]), [], [])

    to._write_unit(tmp_path, unit, "retasy")

    with np.load(tmp_path / "unit.npz") as data:
        assert sorted(data.files) == ["loud.wav", "silent.wav"]  # both are fingerprints
    with np.load(tmp_path / "control" / "unit.npz") as data:
        assert list(data.files) == ["loud.wav"]  # but only one contributes index frames


def test_an_undecodable_frame_row_is_a_fact_but_a_gate_row_is_not(tmp_path, monkeypatch):
    def rows(*items: tuple[bytes, str]) -> None:
        monkeypatch.setattr("tadabur.shard_reader.iter_shard_rows", lambda *args, **kwargs: iter(
            [{"audio": {"bytes": data, "path": path}} for data, path in items]))

    def decode(data: bytes) -> np.ndarray:
        if data == b"broken":
            raise DecodeError("not audio")
        return _signal(3)

    monkeypatch.setattr(to, "decode_to_mono_16k", decode)
    broken: list[str] = []
    rows((b"good", "a.wav"), (b"broken", "b.wav"))

    seen = list(to._frame_unit(0, tmp_path, [], broken, EMPTY_PANEL, {}, REVISION))
    assert [clip for clip, _ in seen] == ["0:0"]
    assert np.array_equal(seen[0][1], _signal(3))
    assert broken == ["0:1"]

    rows((b"broken", "gate.wav"))
    with pytest.raises(ValueError, match="does not decode"):
        list(to._frame_unit(0, tmp_path, [], [], EMPTY_PANEL, {(0, 0): "gate.wav"}, REVISION))

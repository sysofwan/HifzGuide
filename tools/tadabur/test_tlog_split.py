"""Acceptance tests for the tlog clip-level split freeze (``tadabur.tlog_split``, #188).

The owner accepted the S2 fallback on 2026-10-24: the group is the S1 id group, no speaker
embedding or cluster is computed, and ``speaker_cluster`` is a documented ``null`` on every
row. The role is the pre-registered hash of the group key, so the committed freeze must be
reproducible from the committed ``group_key`` alone. The freeze also reports the yield after
the silence floors and each role's surah/ayah distribution against the training pool.

Everything here runs offline: the synthetic build uses tiny index/pass-1 files and a floors
record, never the network, a GPU or the Silero model.

Test command, from the repo root::

    python -m pytest -q tools/tadabur/test_tlog_split.py
"""

from __future__ import annotations

import gzip
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from types import SimpleNamespace

import pytest

#: ``audio_sha256`` of the zero-byte rows unclean-00000 is mostly made of.
EMPTY_SHA256 = hashlib.sha256(b"").hexdigest()
#: Every committed freeze row, and nothing else (``speaker_cluster`` is present and null).
SPLIT_KEYS = frozenset({"split", "shard_file", "row", "audio_path", "speaker_cluster",
                        "group_key", "role"})
ROOT = Path(__file__).resolve().parents[2]
TOTAL_SHARDS = 743
CLEAN_STEM = "clean-00000-of-00411"
UNCLEAN_STEM = "unclean-00000-of-00332"


@pytest.fixture
def ts():
    from tadabur import tlog_split

    return tlog_split


@pytest.fixture
def ti():
    from tadabur import tlog_index

    return tlog_index


@pytest.fixture
def tp():
    from tadabur import tlog_pass1

    return tlog_pass1


def read_jsonl_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def write_jsonl_gz(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def split_files(ts) -> list[Path]:
    return sorted(ts.SPLIT_DIR.glob("*/*.jsonl.gz"))


# --- the synthetic build ---------------------------------------------------------------------


def _index_row(split: str, stem: str, index: int, path: str, label: str | None,
               group: str) -> dict:
    return {"schema_version": 1, "split": split, "shard_file": stem, "row": index,
            "audio_path": path, "filename_label": label, "recording_id": None,
            "id_group": group, "tarteel_v1_eval_overlap": False,
            "tarteel_v1_any_overlap": False}


def _pass1_row(split: str, stem: str, index: int, path: str, *, codec_ok: bool = True,
               ratio: float = 0.9, speech_s: float = 5.0,
               digest: str | None = None) -> dict:
    return {"schema_version": 2, "split": split, "shard_file": stem, "row": index,
            "audio_path": path,
            "audio_sha256": digest or hashlib.sha256(path.encode("utf-8")).hexdigest(),
            "codec_ok": codec_ok, "sample_rate": 16000 if codec_ok else None,
            "duration_s": 10.0 if codec_ok else None,
            "speech_s": speech_s if codec_ok else None,
            "speech_ratio": ratio if codec_ok else None,
            "decoder": "soundfile" if codec_ok else None}


def _synthetic(tmp_path: Path) -> SimpleNamespace:
    index_dir, pass1_dir = tmp_path / "index", tmp_path / "pass1"
    index = {
        "clean": [_index_row("clean", CLEAN_STEM, 0, "1_1_111.flac", "1:1", "1_1_111.flac"),
                  _index_row("clean", CLEAN_STEM, 1, "2_2_222.flac", "2:2", "2_2_222.flac")],
        "unclean": [_index_row("unclean", UNCLEAN_STEM, 0, "3_3_333.wav", "3:3", "3_3_333.wav"),
                    _index_row("unclean", UNCLEAN_STEM, 1, "4_4_444.wav", None, "4_4_444.wav")],
    }
    pass1 = {
        "clean": [_pass1_row("clean", CLEAN_STEM, 0, "1_1_111.flac"),
                  _pass1_row("clean", CLEAN_STEM, 1, "2_2_222.flac", codec_ok=False,
                             digest=EMPTY_SHA256)],
        "unclean": [_pass1_row("unclean", UNCLEAN_STEM, 0, "3_3_333.wav", ratio=0.1),
                    _pass1_row("unclean", UNCLEAN_STEM, 1, "4_4_444.wav", speech_s=0.5)],
    }
    for split, stem in (("clean", CLEAN_STEM), ("unclean", UNCLEAN_STEM)):
        write_jsonl_gz(index_dir / split / f"{stem}.jsonl.gz", index[split])
        write_jsonl_gz(pass1_dir / split / f"{stem}.jsonl.gz", pass1[split])
    floors_path = tmp_path / "silence_floors.json"
    floors_path.write_text(json.dumps({"f_s": 0.4, "d_s": 1.0}), encoding="utf-8")
    return SimpleNamespace(index_dir=index_dir, pass1_dir=pass1_dir, floors=floors_path,
                           index_rows=index["clean"] + index["unclean"])


def test_a_freeze_row_keys_the_id_group_and_the_pre_registered_role(ts, ti):
    row = ts.freeze_row({"split": "clean", "shard_file": "s", "row": 7,
                         "audio_path": "b.flac", "id_group": "a.flac"})

    assert set(row) == SPLIT_KEYS
    assert row == {"split": "clean", "shard_file": "s", "row": 7, "audio_path": "b.flac",
                   "speaker_cluster": None, "group_key": "a.flac",
                   "role": ti.role("a.flac")}


def test_the_build_writes_the_rows_and_reports_the_counts(ts, ti, tmp_path):
    data = _synthetic(tmp_path)
    meta = ts.build(tmp_path / "split", index_dir=data.index_dir, pass1_dir=data.pass1_dir,
                    floors_path=data.floors, code_sha="a" * 40, pull_request=200)

    clean = read_jsonl_gz(tmp_path / "split" / "clean" / f"{CLEAN_STEM}.jsonl.gz")
    unclean = read_jsonl_gz(tmp_path / "split" / "unclean" / f"{UNCLEAN_STEM}.jsonl.gz")
    assert clean == [ts.freeze_row(row) for row in data.index_rows[:2]]
    assert unclean == [ts.freeze_row(row) for row in data.index_rows[2:]]

    assert meta["code_sha"] == "a" * 40
    assert meta["pull_request"] == 200
    assert meta["pull_request_ref"] == "refs/pull/200/head"
    assert meta["embeddings"] == ts.EMBEDDINGS_NOTE == "none (owner fallback 2026-10-24)"
    assert "null" in meta["speaker_cluster"]

    by_role = Counter(ti.role(row["id_group"]) for row in data.index_rows)
    assert meta["counts"]["rows"]["by_role"] == {role: by_role[role] for role in ti.ROLES}
    assert meta["counts"]["rows"]["by_split"] == {"clean": 2, "unclean": 2, "total": 4}
    groups = meta["counts"]["groups"]
    assert groups["count"] == 4
    assert groups["largest"] in {row["id_group"] for row in data.index_rows}
    assert groups["largest_rows"] == 1
    assert groups["largest_share"] == pytest.approx(0.25)

    assert meta["yield"]["clean"] == {"rows": 2, "codec_failed": 1, "empty_bytes": 1,
                                      "silence": 0, "usable": 1, "usable_share": 0.5}
    assert meta["yield"]["unclean"] == {"rows": 2, "codec_failed": 0, "empty_bytes": 0,
                                        "silence": 2, "usable": 0, "usable_share": 0.0}
    assert meta["yield"]["total"]["rows"] == 4

    assert meta["shards"] == {
        f"clean/{CLEAN_STEM}": {"code_sha": "a" * 40, "pull_request": 200},
        f"unclean/{UNCLEAN_STEM}": {"code_sha": "a" * 40, "pull_request": 200},
    }


def test_the_build_is_deterministic(ts, tmp_path):
    data = _synthetic(tmp_path)
    outputs = []
    for name in ("first", "second"):
        target = tmp_path / name
        ts.build(target, index_dir=data.index_dir, pass1_dir=data.pass1_dir,
                 floors_path=data.floors, code_sha="a" * 40, pull_request=200)
        outputs.append(target)

    for path in sorted(outputs[0].glob("*/*.jsonl.gz")):
        assert path.read_bytes() == (outputs[1] / path.relative_to(outputs[0])).read_bytes()
    assert (outputs[0] / "meta.json").read_bytes() == (outputs[1] / "meta.json").read_bytes()


def test_the_build_refuses_pass1_rows_that_disagree_with_the_index(ts, tmp_path):
    data = _synthetic(tmp_path)
    pass1_path = data.pass1_dir / "clean" / f"{CLEAN_STEM}.jsonl.gz"
    write_jsonl_gz(pass1_path, read_jsonl_gz(pass1_path)[:1])

    with pytest.raises(ValueError, match="pass 1"):
        ts.build(tmp_path / "split", index_dir=data.index_dir, pass1_dir=data.pass1_dir,
                 floors_path=data.floors, code_sha="a" * 40, pull_request=200)


def test_the_build_refuses_an_index_that_its_meta_says_is_incomplete(ts, tmp_path):
    data = _synthetic(tmp_path)
    (data.index_dir / "meta.json").write_text(
        json.dumps({"source": {"splits": {"clean": 1, "unclean": 1}}}), encoding="utf-8")
    (data.index_dir / "unclean" / f"{UNCLEAN_STEM}.jsonl.gz").unlink()

    with pytest.raises(ValueError, match="whole index"):
        ts.build(tmp_path / "split", index_dir=data.index_dir, pass1_dir=data.pass1_dir,
                 floors_path=data.floors, code_sha="a" * 40, pull_request=200)


# --- the committed freeze --------------------------------------------------------------------


def test_the_freeze_commits_every_index_row_once(ts, ti):
    files = split_files(ts)
    assert len(files) == TOTAL_SHARDS

    for path in files:
        split, stem = path.parent.name, path.name[: -len(".jsonl.gz")]
        rows = read_jsonl_gz(path)
        index = read_jsonl_gz(ti.INDEX_DIR / split / path.name)
        assert len(rows) == len(index)
        assert [row["row"] for row in rows] == list(range(len(index)))
        assert [row["audio_path"] for row in rows] == [row["audio_path"] for row in index]
        assert all(set(row) == SPLIT_KEYS for row in rows)
        assert {row["split"] for row in rows} == {split}
        assert {row["shard_file"] for row in rows} == {stem}
        assert all(row["speaker_cluster"] is None for row in rows)
        assert all(row["group_key"] == index_row["id_group"]
                   for row, index_row in zip(rows, index))


def test_roles_are_reproducible_from_the_committed_group_keys(ts, ti):
    by_split = Counter()
    by_role = Counter()
    for path in split_files(ts):
        for row in read_jsonl_gz(path):
            assert row["role"] == ti.role(row["group_key"])
            assert row["role"] == ti.role_from_hash(ti.group_hash(row["group_key"]))
            assert row["group_key"] <= row["audio_path"]
            by_split[row["split"]] += 1
            by_role[row["role"]] += 1

    meta = json.loads((ts.SPLIT_DIR / "meta.json").read_text(encoding="utf-8"))
    assert meta["counts"]["rows"]["by_role"] == {role: by_role[role] for role in ti.ROLES}
    assert meta["counts"]["rows"]["by_split"] == {
        **{split: by_split[split] for split in ("clean", "unclean")},
        "total": sum(by_split.values())}
    assert meta["counts"]["rows"]["total"] == 719_853


def test_no_group_exceeds_one_percent_and_the_largest_is_reported(ts):
    groups = Counter()
    for path in split_files(ts):
        for row in read_jsonl_gz(path):
            groups[row["group_key"]] += 1

    total = sum(groups.values())
    largest_key, largest_rows = groups.most_common(1)[0]
    meta = json.loads((ts.SPLIT_DIR / "meta.json").read_text(encoding="utf-8"))
    reported = meta["counts"]["groups"]
    assert largest_rows / total < 0.01
    assert reported["count"] == len(groups)
    assert reported["largest"] == largest_key
    assert reported["largest_rows"] == largest_rows
    assert reported["largest_share"] == pytest.approx(largest_rows / total, abs=1e-6)


def test_the_role_shares_are_the_pre_registered_tranches(ts, ti):
    meta = json.loads((ts.SPLIT_DIR / "meta.json").read_text(encoding="utf-8"))
    total = meta["counts"]["rows"]["total"]

    for role in ti.ROLES:
        assert meta["counts"]["roles"][role]["rows"] == meta["counts"]["rows"]["by_role"][role]
        assert meta["counts"]["roles"][role]["share"] == pytest.approx(
            meta["counts"]["rows"]["by_role"][role] / total, abs=1e-6)
    for role in (ti.LOOK, ti.CONFIRM_1, ti.CONFIRM_2, ti.CONFIRM_3, ti.CONFIRM_4):
        assert 0.015 <= meta["counts"]["roles"][role]["share"] <= 0.025


def test_the_tranche_distribution_is_reported_against_the_training_pool(ts, ti):
    expected = defaultdict(lambda: {"rows": 0, "labelled": 0,
                                    "surah": Counter(), "ayah": Counter()})
    for path in split_files(ts):
        split = path.parent.name
        index = read_jsonl_gz(ti.INDEX_DIR / split / path.name)
        for row, index_row in zip(read_jsonl_gz(path), index):
            bucket = expected[row["role"]]
            bucket["rows"] += 1
            label = index_row["filename_label"]
            if label is not None:
                bucket["labelled"] += 1
                bucket["surah"][label.split(":")[0]] += 1
                bucket["ayah"][label] += 1

    meta = json.loads((ts.SPLIT_DIR / "meta.json").read_text(encoding="utf-8"))
    assert set(meta["distribution"]) == set(ti.ROLES)
    for role in ti.ROLES:
        reported = meta["distribution"][role]
        assert reported["rows"] == expected[role]["rows"]
        assert reported["labelled"] == expected[role]["labelled"]
        assert reported["surah"] == dict(expected[role]["surah"])
        assert reported["ayah"] == dict(expected[role]["ayah"])
        assert sum(reported["surah"].values()) == reported["labelled"]
        assert sum(reported["ayah"].values()) == reported["labelled"]
    assert meta["distribution"][ti.TRAIN]["rows"] > 0


def test_the_yield_report_uses_the_shared_floors_reader(ts, tp):
    floors = json.loads((ROOT / "tools/tadabur/tlog/silence_floors.json")
                        .read_text(encoding="utf-8"))
    expected = defaultdict(Counter)
    for path in sorted(tp.PASS1_DIR.glob("*/*.jsonl.gz")):
        split = path.parent.name
        for row in read_jsonl_gz(path):
            reasons = tp.exclusion_reasons(row, floors)
            expected[split]["rows"] += 1
            expected[split]["codec_failed"] += "codec" in reasons
            expected[split]["empty_bytes"] += ("codec" in reasons
                                               and row["audio_sha256"] == EMPTY_SHA256)
            expected[split]["silence"] += "silence" in reasons
            expected[split]["usable"] += not reasons

    meta = json.loads((ts.SPLIT_DIR / "meta.json").read_text(encoding="utf-8"))
    for split in ("clean", "unclean"):
        reported = meta["yield"][split]
        assert reported["rows"] == expected[split]["rows"]
        assert reported["codec_failed"] == expected[split]["codec_failed"]
        assert reported["empty_bytes"] == expected[split]["empty_bytes"]
        assert reported["silence"] == expected[split]["silence"]
        assert reported["usable"] == expected[split]["usable"]
        assert reported["usable_share"] == pytest.approx(
            expected[split]["usable"] / expected[split]["rows"], abs=1e-6)
    assert meta["yield"]["total"]["rows"] == sum(
        expected[split]["rows"] for split in ("clean", "unclean"))


def test_the_meta_cites_the_frozen_inputs_and_the_owner_fallback(ts):
    meta = json.loads((ts.SPLIT_DIR / "meta.json").read_text(encoding="utf-8"))
    prereg = ROOT / "docs/tlog-holdout-preregistration.md"
    report = ROOT / "tools/tadabur/tlog/s2/report.json"
    pass1_meta = ROOT / "tools/tadabur/tlog/pass1/meta.json"
    floors = ROOT / "tools/tadabur/tlog/silence_floors.json"

    assert re.fullmatch(r"[0-9a-f]{40}", meta["code_sha"])
    assert isinstance(meta["pull_request"], int) and meta["pull_request"] > 0
    assert meta["pull_request_ref"] == f"refs/pull/{meta['pull_request']}/head"
    assert meta["embeddings"] == "none (owner fallback 2026-10-24)"
    assert "null" in meta["speaker_cluster"]
    assert meta["s2_report"] == {
        "path": "tools/tadabur/tlog/s2/report.json",
        "sha256": hashlib.sha256(report.read_bytes()).hexdigest()}
    assert meta["preregistration"] == {
        "path": "docs/tlog-holdout-preregistration.md",
        "sha256": hashlib.sha256(prereg.read_bytes()).hexdigest()}
    assert meta["pass1"] == {
        "path": "tools/tadabur/tlog/pass1",
        "meta_sha256": hashlib.sha256(pass1_meta.read_bytes()).hexdigest()}
    floors_record = json.loads(floors.read_text(encoding="utf-8"))
    assert meta["floors"] == {
        "path": "tools/tadabur/tlog/silence_floors.json",
        "sha256": hashlib.sha256(floors.read_bytes()).hexdigest(),
        "f_s": floors_record["f_s"], "d_s": floors_record["d_s"]}


def test_the_meta_names_each_shard_s_producing_sha_and_pr(ts):
    meta = json.loads((ts.SPLIT_DIR / "meta.json").read_text(encoding="utf-8"))
    shards = meta["shards"]
    tracer = {f"clean/{CLEAN_STEM}", "clean/clean-00205-of-00411", f"unclean/{UNCLEAN_STEM}"}
    assert len(shards) == TOTAL_SHARDS

    for key in tracer:
        assert shards[key]["code_sha"].startswith("d2af0e3")
        assert shards[key]["pull_request"] == 198
    for key, entry in shards.items():
        if key in tracer:
            continue
        assert re.fullmatch(r"[0-9a-f]{40}", entry["code_sha"])
        assert entry["pull_request"] == meta["pull_request"]

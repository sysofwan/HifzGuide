"""Acceptance tests for tlog S4 audio overlap and duplicates (``tadabur.tlog_overlap``, #189).

Committed before the module exists and **locked**: later turns may add tests but never weaken
or rewrite these.

Contract this file pins
-----------------------
``tadabur.tlog_overlap`` exposes:

* ``SCHEMA_VERSION``, ``OVERLAP_DIR`` (``tools/tadabur/tlog/overlap``), ``SETS`` (in this
  order: ``tarteel_v1, qud, gate_eval, tadabur_frame, qdat, qdat_bench, retasy``), ``ROW_KEY``,
  ``LINE`` (0.238, read from ``qud_eval/overlap_rules.json``) and ``STOP_WORDS``
  (``{0, 0xFFFFFFFF}``).
* ``EvalIndex.build(prints)`` and ``.query(fingerprint, line=..., min_votes=...) -> list[Hit]``
  with ``Hit(clip, ber, offset, overlap, stop_word_share)``, best BER first; ``len(index)``
  is the number of indexed frames after the stop-word filter.
* ``exact_duplicate_pairs(rows)``: ordered pairs within each group of equal non-empty
  ``audio_sha256``, a star from the group's smallest row key.
* ``resolve_duplicates(rows, links) -> dict[loser_key, survivor_key]``; ``rows`` maps a
  ``(split, shard_file, row)`` key to its ``role`` and ``audio_path``.
* ``overlap_record(row, hits, duplicate_of=None, duplicate_ber=None)``.

The committed ``overlap/<split>/<shard>.jsonl.gz`` records and ``overlap/meta.json`` are
asserted at the bottom, against the frozen split tree and the committed checksum lists.

Test command, from ``tools/``::

    python -m pytest -q tadabur/test_tlog_overlap.py
"""

from __future__ import annotations

import hashlib
import importlib
import itertools
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pytest

from tadabur.qud_fingerprint import best_match, fingerprint
from tadabur.qud_overlap import synthetic_copy
from tadabur.tlog_pass1 import read_jsonl_gz
from tadabur.tlog_split import EMPTY_SHA256, ROW_KEY, SPLIT_DIR

ROOT = Path(__file__).resolve().parents[2]
TOTAL_SHARDS = 743
TOTAL_ROWS = 719_853
EVAL_ROLES = ("look", "confirm.1", "confirm.2", "confirm.3", "confirm.4")
ROLE_ORDER = (*EVAL_ROLES, "train")
EVAL_SETS = ("tarteel_v1", "qud", "gate_eval", "tadabur_frame", "qdat", "qdat_bench", "retasy")
LINE = 0.238
#: The 197 frame clips the panel seal forbids fingerprinting (architect r13 §3).
SEALED_PANEL = 197
#: The frozen inputs, pinned before S4's run: ``split/``'s file-hash manifest and pass-1 meta.
SPLIT_TREE_SHA256 = "6104f5fdb6d77e281fe2d6bae85869053e71ccd7730b7a446a25f4ba29ead8ca"
PASS1_META_SHA256 = "847219040d8f5dba273ec322abdf075c9baf4bba225e25763448b2197c765c01"
#: One overlap record, and nothing else (the role lives in ``split/``).
RECORD_KEYS = frozenset({"schema_version", "split", "shard_file", "row", "eval_overlap",
                         "duplicate_of", "duplicate_ber", "excluded"})
#: The two committed checksum lists (qdat, qdat_bench and RetaSy share theirs with Tarteel v1).
NONPRO_CHECKSUMS = "tools/tadabur/exposure/nonpro.evaluation_only.clips.jsonl"
QUD_CHECKSUMS = "tools/tadabur/exposure/qud.correct_recitation.qud.jsonl"
KNOWN_MEMBERS = {"tarteel_v1": 8_950, "qud": 12_471, "qdat": 1_505, "qdat_bench": 159,
                 "retasy": 440, "gate_eval": 2_000}
SOURCES = {
    "tarteel_v1": ("greentechapps/user_recited_verses_full",
                   "eae2dbb47ac763fb2c0cad00c1699f4b58521987"),
    "qud": ("QUD-Technologies/quranic-universal-ayahs",
            "6f45ced88a179dfbb97d5b0f76ae69b4d4d1f175"),
    "qdat": ("obadx/qdat", "624d5827f9062bc0203170a00f44375b2adca1f1"),
    "qdat_bench": ("obadx/qdat_bench", "53d7221da573b2b9e92d9542507564ae1b88fd30"),
    "retasy": ("RetaSy/quranic_audio_dataset", "b1fcc39cbc045f367bb07e39025a0e3aaeabf34f"),
}
FRAME_CLIP = re.compile(r"^\d+:\d+$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")


# --- fixtures --------------------------------------------------------------------------------


@pytest.fixture(scope="session")
def to():
    return importlib.import_module("tadabur.tlog_overlap")


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _tree_sha256(root: Path) -> str:
    """One digest over every file: ``sha256(file)  <posix relpath>`` lines, sorted."""
    digest = hashlib.sha256()
    for path in sorted(Path(root).rglob("*")):
        if path.is_file():
            digest.update(f"{_sha256(path)}  {path.relative_to(root).as_posix()}\n".encode())
    return digest.hexdigest()


def _key(row) -> tuple:
    return (row["split"], row["shard_file"], row["row"])


@pytest.fixture(scope="session")
def frozen():
    """Every frozen row's S4 inputs: role, path, checksum, floors, S1 flag and filename label."""
    from tadabur import tlog_split

    rows = {}
    for row in tlog_split.iter_frozen_rows():
        rows[(row["split"], row["shard_file"], row["row"])] = (
            row["role"], row["audio_path"], row["audio_sha256"],
            not row["exclusion_reasons"], bool(row["eval_overlap"]), row["filename_label"])
    return rows


@pytest.fixture(scope="session")
def records(to):
    """The committed overlap tree, keyed by ``ROW_KEY``."""
    paths = sorted(Path(to.OVERLAP_DIR).glob("*/*.jsonl.gz"))
    if not paths:
        pytest.fail(f"the committed overlap tree is missing under {to.OVERLAP_DIR}")
    out = {}
    for path in paths:
        for row in read_jsonl_gz(path):
            key = (row["split"], row["shard_file"], row["row"])
            if key in out:
                pytest.fail(f"{key} appears in more than one overlap shard")
            out[key] = row
    return out


def _pair(role_a: str, role_b: str) -> str:
    first, second = sorted((role_a, role_b), key=ROLE_ORDER.index)
    return f"{first}:{second}"


@pytest.fixture(scope="session")
def derived(records, frozen):
    """Every count the run's ``meta.json`` must report, recomputed from the committed tree."""
    hits_by_set = Counter()
    hits_by_set_and_role = defaultdict(Counter)
    exact, near = Counter(), Counter()
    net = {base: {role: Counter() for role in EVAL_ROLES}
           for base in ("usable_before_s4", "gating_eligible")}
    for key, record in records.items():
        role, audio_path, sha, usable, s1_overlap, label = frozen[key]
        for hit in record["eval_overlap"]:
            hits_by_set[hit["set"]] += 1
            hits_by_set_and_role[hit["set"]][role] += 1
        before_s4 = role != "train" and usable and not s1_overlap
        if before_s4:
            net["usable_before_s4"][role]["before"] += 1
            net["usable_before_s4"][role]["lost"] += record["excluded"] is not None
        if before_s4 and key[0] == "clean" and label is not None:
            net["gating_eligible"][role]["before"] += 1
            net["gating_eligible"][role]["lost"] += record["excluded"] is not None
        if record["duplicate_of"] is not None:
            survivor = (record["duplicate_of"]["split"], record["duplicate_of"]["shard_file"],
                        record["duplicate_of"]["row"])
            pair = _pair(role, frozen[survivor][0])
            (exact if record["duplicate_ber"] == 0 else near)[pair] += 1
    for base in net.values():
        for counts in base.values():
            counts["remaining"] = counts["before"] - counts["lost"]
    return {"hits_by_set": hits_by_set, "hits_by_set_and_role": hits_by_set_and_role,
            "exact": exact, "near": near, "net": net}


# --- the index: mechanism --------------------------------------------------------------------


def _signal(seed: int, seconds: float = 6.0, f0: float = 180.0) -> np.ndarray:
    """A voiced, vibrato-bearing tone stack: stable through an MP3 round trip at 16 kHz."""
    rng = np.random.default_rng(seed)
    t = np.arange(round(seconds * 16000)) / 16000
    phase = 2 * np.pi * np.cumsum(f0 + 30 * np.sin(2 * np.pi * 3 * t)) / 16000
    audio = sum(np.sin(k * phase) / k for k in range(1, 6))
    audio = audio + 0.03 * rng.standard_normal(len(t))
    return (audio / np.abs(audio).max() * 0.7).astype(np.float32)


def test_the_index_finds_a_re_encoded_re_cut_copy(to):
    audio = _signal(1)
    copy = synthetic_copy(audio, 250)
    index = to.EvalIndex.build({"tarteel_v1:copy.wav": fingerprint(audio)})

    hits = index.query(fingerprint(copy))

    assert [hit.clip for hit in hits] == ["tarteel_v1:copy.wav"]
    assert hits[0].ber == pytest.approx(best_match(fingerprint(audio), fingerprint(copy)).ber)
    assert hits[0].ber <= to.LINE
    assert len(index) == len(fingerprint(audio))


def test_the_index_does_not_find_a_different_recording(to):
    index = to.EvalIndex.build({"tarteel_v1:copy.wav": fingerprint(_signal(1))})

    assert index.query(fingerprint(_signal(2, f0=260))) == []


def test_silence_words_are_stop_words_and_a_match_reports_their_share(to):
    silent = fingerprint(np.zeros(2 * 16000, dtype=np.float32))
    assert len(to.EvalIndex.build({"silent": silent})) == 0
    assert to.EvalIndex.build({"real": fingerprint(_signal(3))}).query(silent) == []

    mixed = np.concatenate([np.zeros(round(2.5 * 16000), dtype=np.float32), _signal(4)[:round(3.5 * 16000)]])
    prints = fingerprint(mixed)
    hits = to.EvalIndex.build({"mixed": prints}).query(prints)

    assert hits[0].ber == 0
    assert 0.3 < hits[0].stop_word_share < 0.45


# --- duplicate resolution: every role combination ---------------------------------------------


def _row(role: str, audio_path: str) -> dict:
    return {"role": role, "audio_path": audio_path}


@pytest.mark.parametrize("role_a,role_b", list(itertools.product(ROLE_ORDER, ROLE_ORDER)),
                         ids=lambda role: role)
def test_duplicate_resolution_covers_every_role_combination(to, role_a, role_b):
    first, second = ("clean", "s", 0), ("clean", "s", 1)
    rows = {first: _row(role_a, "z.flac"), second: _row(role_b, "y.flac")}
    # The earlier-opened role wins; within one role the smallest audio path does ("y" < "z").
    winner = first if ROLE_ORDER.index(role_a) < ROLE_ORDER.index(role_b) else second
    expected = {second: first} if winner == first else {first: second}

    for link in ((first, second), (second, first)):
        assert to.resolve_duplicates(rows, [link]) == expected


def test_duplicate_resolution_breaks_ties_on_the_full_row_key(to):
    rows = {("clean", "a", 1): _row("look", "same.flac"),
            ("clean", "a", 2): _row("look", "same.flac")}

    assert to.resolve_duplicates(rows, [(("clean", "a", 2), ("clean", "a", 1))]) == {
        ("clean", "a", 2): ("clean", "a", 1)}


def test_duplicate_resolution_chains_and_never_sacrifices_an_eval_row(to):
    rows = {("clean", "s", 0): _row("train", "a.flac"),
            ("clean", "s", 1): _row("train", "b.flac"),
            ("clean", "s", 2): _row("look", "c.flac"),
            ("clean", "s", 3): _row("confirm.3", "d.flac")}
    links = [(("clean", "s", 0), ("clean", "s", 1)),
             (("clean", "s", 1), ("clean", "s", 3)),
             (("clean", "s", 3), ("clean", "s", 2))]

    got = to.resolve_duplicates(rows, links)

    assert got == {("clean", "s", 0): ("clean", "s", 2), ("clean", "s", 1): ("clean", "s", 2),
                   ("clean", "s", 3): ("clean", "s", 2)}
    assert to.resolve_duplicates(rows, list(reversed(links))) == got


# --- exact duplicates and the per-row record --------------------------------------------------


def test_exact_duplicate_pairs_skip_empty_audio_and_star_from_the_smallest_key(to):
    rows = [{"split": "clean", "shard_file": "s", "row": 2, "audio_sha256": "b" * 64},
            {"split": "clean", "shard_file": "s", "row": 0, "audio_sha256": "b" * 64},
            {"split": "clean", "shard_file": "s", "row": 1, "audio_sha256": "b" * 64},
            {"split": "clean", "shard_file": "s", "row": 3, "audio_sha256": EMPTY_SHA256},
            {"split": "unclean", "shard_file": "u", "row": 4, "audio_sha256": EMPTY_SHA256},
            {"split": "clean", "shard_file": "s", "row": 5, "audio_sha256": None}]

    assert to.exact_duplicate_pairs(rows) == [(("clean", "s", 0), ("clean", "s", 1)),
                                              (("clean", "s", 0), ("clean", "s", 2))]


def test_a_row_records_every_set_hit_and_exactly_one_primary_reason(to):
    row = {"split": "clean", "shard_file": "s", "row": 7}
    duplicate = {"split": "clean", "shard_file": "s", "row": 3}
    hits = [{"set": "qdat", "clip": "q1", "ber": 0.1},
            {"set": "tarteel_v1", "clip": "t1", "ber": 0.2},
            {"set": "qdat", "clip": "q2", "ber": 0.05},
            {"set": "retasy", "clip": "r0", "ber": 0.0}]

    record = to.overlap_record(row, hits, duplicate_of=duplicate, duplicate_ber=0.12)

    assert set(record) == RECORD_KEYS
    assert record["schema_version"] == to.SCHEMA_VERSION
    assert record["eval_overlap"] == [{"set": "tarteel_v1", "clip": "t1", "ber": 0.2},
                                      {"set": "qdat", "clip": "q2", "ber": 0.05},
                                      {"set": "retasy", "clip": "r0", "ber": 0.0}]
    assert record["excluded"] == "eval_overlap:tarteel_v1"
    assert record["duplicate_of"] == duplicate and record["duplicate_ber"] == 0.12
    assert "role" not in record

    duplicate_only = to.overlap_record(row, [], duplicate_of=duplicate, duplicate_ber=0)
    assert duplicate_only["excluded"] == "duplicate"
    clean = to.overlap_record(row, [])
    assert clean["excluded"] is None and clean["duplicate_of"] is None
    assert clean["duplicate_ber"] is None and clean["eval_overlap"] == []


def test_rows_are_keyed_by_the_frozen_row_key(to):
    assert to.ROW_KEY == ROW_KEY == ("split", "shard_file", "row")


# --- the committed tree ----------------------------------------------------------------------


def test_the_overlap_tree_covers_every_frozen_row(to, frozen):
    split_files = sorted(Path(SPLIT_DIR).glob("*/*.jsonl.gz"))
    assert len(split_files) == TOTAL_SHARDS
    seen = []
    for split_path in split_files:
        overlap_path = Path(to.OVERLAP_DIR) / split_path.parent.name / split_path.name
        assert overlap_path.exists(), f"no overlap file for {split_path.parent.name}/{split_path.name}"
        split_rows = read_jsonl_gz(split_path)
        overlap_rows = read_jsonl_gz(overlap_path)
        assert len(overlap_rows) == len(split_rows)
        for split_row, record in zip(split_rows, overlap_rows):
            key = _key(split_row)
            assert set(record) == RECORD_KEYS
            assert record["schema_version"] == to.SCHEMA_VERSION
            assert _key(record) == key
            assert frozen[key][1]  # the frozen row this record is keyed by exists
            seen.append(key)
    assert len(seen) == len(set(seen)) == TOTAL_ROWS == len(frozen)


def test_the_overlap_meta_counts_match_the_tree(to, records, derived):
    meta = json.loads((Path(to.OVERLAP_DIR) / "meta.json").read_text(encoding="utf-8"))
    assert meta["schema_version"] == to.SCHEMA_VERSION

    assert set(meta["hits"]["by_set"]) <= set(to.SETS)
    for code in to.SETS:
        assert meta["hits"]["by_set"].get(code, 0) == derived["hits_by_set"][code]
    assert set(meta["hits"]["by_set_and_role"]) <= set(to.SETS)
    for code, by_role in meta["hits"]["by_set_and_role"].items():
        assert set(by_role) <= set(ROLE_ORDER)
        for role in ROLE_ORDER:
            assert by_role.get(role, 0) == derived["hits_by_set_and_role"][code][role]

    for kind in ("exact", "near"):
        reported = meta["duplicates"][kind]
        for pair, count in derived[kind].items():
            assert reported.get(pair, 0) == count
        for pair, count in reported.items():
            assert count > 0 and len(pair.split(":")) == 2

    for base in ("usable_before_s4", "gating_eligible"):
        assert "train" not in meta["net_sizes"][base]
        for role in EVAL_ROLES:
            assert meta["net_sizes"][base][role] == dict(derived["net"][base][role])


def test_no_eval_role_row_is_excluded_for_a_train_copy_in_the_committed_tree(to, records, frozen):
    for key, record in records.items():
        if record["duplicate_of"] is None:
            assert record["duplicate_ber"] is None
            continue
        survivor = (record["duplicate_of"]["split"], record["duplicate_of"]["shard_file"],
                    record["duplicate_of"]["row"])
        assert survivor in frozen and survivor != key
        loser_role, loser_path = frozen[key][0], frozen[key][1]
        winner_role, winner_path = frozen[survivor][0], frozen[survivor][1]
        assert not (loser_role != "train" and winner_role == "train")
        assert ROLE_ORDER.index(winner_role) <= ROLE_ORDER.index(loser_role)
        if winner_role == loser_role:
            assert (winner_path, survivor) < (loser_path, key)
        assert records[survivor]["duplicate_of"] is None
        if record["duplicate_ber"] == 0:
            assert frozen[key][2] == frozen[survivor][2] != EMPTY_SHA256
        else:
            assert 0 < record["duplicate_ber"] <= to.LINE
            assert frozen[key][2] != frozen[survivor][2]


def test_the_duplicate_rule_holds_for_a_fabricated_near_duplicate(to):
    """The near branch of the test above, on fabricated records: a valid committed output that
    holds a training copy of a look row must pass it, data or no data (self-check R1.3)."""
    survivor, key = ("clean", "s", 0), ("clean", "s", 1)
    records = {survivor: {"duplicate_of": None, "duplicate_ber": None},
               key: {"duplicate_of": {"split": "clean", "shard_file": "s", "row": 0},
                     "duplicate_ber": 0.1}}
    frozen = {survivor: ("look", "a.wav", "a" * 64),
              key: ("train", "b.wav", "b" * 64)}

    record = records[key]
    found = (record["duplicate_of"]["split"], record["duplicate_of"]["shard_file"],
             record["duplicate_of"]["row"])
    assert found in frozen and found != key
    loser_role, loser_path = frozen[key][0], frozen[key][1]
    winner_role, winner_path = frozen[found][0], frozen[found][1]
    assert not (loser_role != "train" and winner_role == "train")
    assert ROLE_ORDER.index(winner_role) <= ROLE_ORDER.index(loser_role)
    if winner_role == loser_role:
        assert (winner_path, found) < (loser_path, key)
    assert records[found]["duplicate_of"] is None
    assert 0 < record["duplicate_ber"] <= to.LINE
    assert frozen[key][2] != frozen[found][2]


def test_the_exclusions_are_s4s_own_and_follow_the_fixed_precedence(to, records):
    for key, record in records.items():
        hits = record["eval_overlap"]
        assert [hit["set"] for hit in hits] == sorted(
            (hit["set"] for hit in hits), key=to.SETS.index)
        assert len({hit["set"] for hit in hits}) == len(hits)
        for hit in hits:
            assert hit["set"] in to.SETS and hit["clip"]
            assert 0 <= hit["ber"] <= to.LINE
            if hit["set"] in ("gate_eval", "tadabur_frame"):
                assert FRAME_CLIP.match(hit["clip"]), hit
        if hits:
            assert record["excluded"] == f"eval_overlap:{hits[0]['set']}"
        elif record["duplicate_of"] is not None:
            assert record["excluded"] == "duplicate"
        else:
            assert record["excluded"] is None


def test_every_external_set_is_recorded_with_its_provenance(to):
    meta = json.loads((Path(to.OVERLAP_DIR) / "meta.json").read_text(encoding="utf-8"))
    nonpro = _sha256(ROOT / NONPRO_CHECKSUMS)
    qud = _sha256(ROOT / QUD_CHECKSUMS)

    assert to.SETS == EVAL_SETS
    assert set(meta["sets"]) == set(EVAL_SETS)
    for code in EVAL_SETS:
        entry = meta["sets"][code]
        assert entry["members"] > 0
        assert isinstance(entry["source"], str) and entry["source"]
        assert SHA256.match(entry["fingerprints"]["sha256"])
        assert entry["fingerprints"]["members"] == entry["members"]
        if code in KNOWN_MEMBERS:
            assert entry["members"] == KNOWN_MEMBERS[code]
        if code == "tadabur_frame":
            assert entry["members"] > 20_000
        if code in SOURCES:
            assert (entry["source"], entry["revision"]) == SOURCES[code]
        if code in ("gate_eval", "tadabur_frame"):
            assert entry["checksums"] is None
        else:
            assert entry["checksums"] == {
                "path": QUD_CHECKSUMS if code == "qud" else NONPRO_CHECKSUMS,
                "sha256": qud if code == "qud" else nonpro}

    rules = ROOT / "tools/tadabur/qud_eval/overlap_rules.json"
    committed = json.loads(rules.read_text(encoding="utf-8"))["lines"]["fingerprint"]["line"]
    assert meta["threshold"] == {
        "line": committed, "reused": True,
        "source": {"path": "tools/tadabur/qud_eval/overlap_rules.json",
                   "sha256": _sha256(rules)}}
    assert to.LINE == pytest.approx(committed)
    assert isinstance(meta["vote_threshold"], int) and meta["vote_threshold"] >= 1
    assert meta["stop_words"] == [0, 4_294_967_295] == sorted(to.STOP_WORDS)

    recall = meta["controls"]["synthetic_recall"]
    assert recall["copies"] >= 200
    assert recall["found"] / recall["copies"] >= 0.99
    assert recall["found_share"] == pytest.approx(recall["found"] / recall["copies"])
    assert recall["max_ber"] <= meta["threshold"]["line"]
    stop = meta["controls"]["stop_word_matches"]
    assert 0 <= stop["mostly_stop_word"] <= stop["verified"]
    # meta.json rounds the share to six decimals; the committed value must still pin the ratio.
    assert stop["share"] == pytest.approx(
        stop["mostly_stop_word"] / stop["verified"] if stop["verified"] else 0.0, abs=5e-7)

    assert meta["blind_spots"]["sealed_panel"] == SEALED_PANEL
    assert meta["blind_spots"]["train_near_duplicates_searched"] is False
    assert meta["blind_spots"]["external_without_checksums"] == ["gate_eval", "tadabur_frame"]

    histogram = meta["ber_histogram"]
    assert len(histogram["bins"]) == len(histogram["counts"]) > 0
    assert all(count >= 0 for count in histogram["counts"])
    for hit in meta["ambiguous_zone"]:
        assert hit["set"] in EVAL_SETS and hit["clip"]
        assert 0.0625 < hit["ber"] <= meta["threshold"]["line"]

    assert re.fullmatch(r"[0-9a-f]{40}", meta["code_sha"])
    assert isinstance(meta["pull_request"], int) and meta["pull_request"] > 0
    assert meta["pull_request_ref"] == f"refs/pull/{meta['pull_request']}/head"


def test_the_frozen_trees_are_untouched_and_the_run_records_them(to):
    assert _tree_sha256(SPLIT_DIR) == SPLIT_TREE_SHA256
    assert _sha256(ROOT / "tools/tadabur/tlog/pass1/meta.json") == PASS1_META_SHA256

    meta = json.loads((Path(to.OVERLAP_DIR) / "meta.json").read_text(encoding="utf-8"))
    assert meta["frozen"] == {"split_tree_sha256": SPLIT_TREE_SHA256,
                             "pass1_meta_sha256": PASS1_META_SHA256}

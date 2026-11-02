"""Phase B of #263 (the confirm.1/2 pool extension): rows, pin, decode batch, merge, driver.

Phase B decodes the **labelled ``reported:*`` rows** of the two mined tranches — straddles,
identified, below-bar and unclean-identified — with the teacher, C95 and round 2's EMA candidate
in the ``checks`` and ``spans`` stores. Everything here is torch-free. The launcher is
``test_tlog_mining_phase_b_launcher.py``.

Contract this file pins
-----------------------
``tadabur.tlog_mining.selection``

* ``MINING_AMENDMENT_PATH`` / ``MINING_AMENDMENT_SHA256``: ``docs/tlog-mining-amendment.md`` and
  the SHA-256 of its bytes. ``require_phase(phase, *, amendment_path=MINING_AMENDMENT_PATH)``:
  ``"A"`` passes; ``"B"`` passes only when the file at ``amendment_path`` hashes to the pin (a
  missing or changed file is a ``ValueError`` naming the amendment); ``"C"`` is always a
  ``ValueError`` naming ``phase C`` and ``round 3``. The committed amendment's "Pending" line is
  replaced by one line recording the Fable AGREE cross-check
  (https://github.com/sysofwan/HifzGuide/issues/262#issuecomment-6588663912) and the pin is that
  edited file's hash.
* ``phase_b_rows(tranche, *, tranches_dir, split_dir, layout, spec, phonetizable=None)
  -> (decodable, unphonetizable)``: two lists of manifest rows of ``confirm.1`` or ``confirm.2``.
  A row is selected when its stratum is ``reported:straddles``, ``reported:identified`` or
  ``reported:below_bar``, or ``reported:unclean`` with a ``label_source`` other than ``none``;
  gating rows (phase A's) and unlabelled unclean rows are never selected. ``phonetizable(label)``
  is called with the row's **``assigned_label``** (the reference the row will be read against,
  not its filename) and a row it rejects goes to ``unphonetizable`` instead of ``decodable``
  (#248). ``None`` means the real phonetizer. Every other tranche (``look``, ``confirm.3``,
  ``confirm.4``, ``train``, anything) is a ``ValueError`` before a file opens; the claim, the
  committed choice and the split-tree role checks of ``phase_a_rows`` apply unchanged.

``tadabur.tlog_mining.mined_decode``

* ``MINED_MODELS == ("base", "c95", "r2_candidate_ema")``; ``MINED_STORES`` holds exactly
  ``checks`` (``truth_baseline.CHECK_STORE``) and ``spans`` (a store whose one protocol is
  ``truth_baseline.PROTOCOLS["spans"]`` and whose decode is ``decoder.decode_spans([span])[0]``
  under the key ``"spans"``).
* ``decode_batch(rows, parts_dir, *, role, decoders, stream, word_counts, budget_seconds, clock,
  share_features, load_audio=None, clip_cache=None, stores=MINED_STORES) -> BatchResult``: one
  tranche's rows, shard by shard in sorted order, through ``tlog_eval_decode.run_shard`` (parts
  at ``tlog_eval_decode.decodes_path(parts_dir, shard, model, role)``, resume by key). The budget
  is checked **before each row**, and a row is atomic across models and stores (every model's
  line lands together), so a started row is never cut off and a resumed batch repeats no row;
  ``done`` is true when every row holds every model.

``tadabur.tlog_mining.merge.merge_mined_decodes(parts_dir, out_dir, *, rows, unphonetizable,
provenance, models) -> {model: line count}``: ``rows`` and ``unphonetizable`` map a tranche to
its decodable and listed rows. Writes ``tlog.mined.<model>.jsonl.gz`` (sorted by key; every line
gains ``tranche`` and the manifest row's ``label_source``) and ``meta.json``.

``tadabur.tlog_mining.__main__.run`` with ``phase="B"`` takes the same arguments as phase A.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tadabur import tlog_eval_decode as ted
from tadabur.test_tlog_eval_confirm import SHARDS, _claim, _manifests
from tadabur.test_tlog_eval_decode import _manifest_row, _write_jsonl
from tadabur.test_tlog_mining_selection import _split_tree
from tadabur.tlog_mining import __main__ as cli
from tadabur.tlog_mining import mined_decode, selection
from tadabur.tlog_mining.batch import BatchResult
from tadabur.tlog_mining.give_way import GiveWayResult
from tadabur.tlog_mining.merge import merge_mined_decodes
from tadabur.tlog_mining.selection import phase_b_rows, require_phase
from tadabur.tlog_pass1 import read_jsonl_gz
from training import round2_verdict_fixtures as rv
from training import truth_baseline
from training.nonpro_decode import item_key

REPO = Path(__file__).resolve().parents[2]
AMENDMENT = REPO / "docs" / "tlog-mining-amendment.md"
CROSS_CHECK = "https://github.com/sysofwan/HifzGuide/issues/262#issuecomment-6588663912"
MINED = ("confirm.1", "confirm.2")
ROLES = ("look", "confirm.1", "confirm.2", "confirm.3", "confirm.4")
MODELS = ("base", "c95", "r2_candidate_ema")
#: #248's phonetizer refusals.
UNPHONETIZABLE_AYAT = {"55:17", "90:8", "90:9", "90:10", "106:1", "106:2", "106:3", "106:4"}


def _phonetizable(label: str) -> bool:
    return label not in UNPHONETIZABLE_AYAT


def _tranche(role: str) -> list[dict]:
    """Rows 1-2 gating; 3-6 the four mineable reported strata; 7 unlabelled unclean; 8 listed."""
    shard = SHARDS[role]

    def row(number, stratum, **overrides):
        # A clip belongs to one tranche: its audio path is unique across them.
        return _manifest_row(number, shard_file=shard, stratum=stratum,
                             audio_path=f"{role}_{number}.flac", **overrides)

    return [row(1, "gating"), row(2, "gating"),
            row(3, "reported:straddles", straddles=True, label_source="identified",
                assigned_label="2:2"),
            row(4, "reported:below_bar", label_source="filename"),
            row(5, "reported:identified", label_source="identified", assigned_label="2:2"),
            row(6, "reported:unclean", split="unclean", label_source="identified",
                assigned_label="2:2"),
            row(7, "reported:unclean", split="unclean", label=None),
            row(8, "reported:identified", label_source="identified", assigned_label="55:17")]


def _decodable(role: str) -> list[dict]:
    return [row for row in _tranche(role) if row["row"] in (3, 4, 5, 6)]


def _claimed(tmp_path, manifests: dict[str, list[dict]], *, claims=MINED):
    """A round with ``manifests`` committed, ``claims`` committed, and its split tree."""
    round_ = rv.committed_round(tmp_path)
    _manifests(round_, manifests)
    for role in claims:
        _claim(round_, role)
    split = _split_tree(tmp_path / "split",
                        {role: manifests.get(role, []) for role in ROLES})
    return round_, split


@pytest.fixture
def mined(tmp_path):
    """Both claims, the committed choice and manifests with every stratum, plus the split tree."""
    return _claimed(tmp_path, {role: _tranche(role) for role in MINED})


def _select(fixture, tranche, **kwargs):
    round_, split = fixture
    kwargs.setdefault("phonetizable", _phonetizable)
    return phase_b_rows(tranche, tranches_dir=round_.tranches, split_dir=split,
                        layout=round_.layout, spec=round_.spec, **kwargs)


def _rows_of(rows) -> list[int]:
    return sorted(row["row"] for row in rows)


# --- the amendment pin -----------------------------------------------------------------------


def test_the_amendment_records_the_cross_check_and_the_pin_is_its_hash():
    text = AMENDMENT.read_text(encoding="utf-8")
    assert CROSS_CHECK in text
    assert "**Pending:**" not in text
    assert selection.MINING_AMENDMENT_PATH == AMENDMENT
    assert selection.MINING_AMENDMENT_SHA256 == hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()


def test_phase_b_runs_under_the_committed_amendment():
    require_phase("B")


def test_phase_b_is_refused_for_a_changed_or_missing_amendment(tmp_path):
    changed = tmp_path / "amendment.md"
    changed.write_bytes(AMENDMENT.read_bytes() + b"\nA late edit.\n")
    for path in (changed, tmp_path / "absent.md"):
        with pytest.raises(ValueError, match="amendment") as raised:
            require_phase("B", amendment_path=path)
        assert "SHA-256" in str(raised.value)


def test_phase_c_is_refused_whatever_the_amendment():
    with pytest.raises(ValueError, match="phase C") as raised:
        require_phase("C", amendment_path=AMENDMENT)
    assert "round 3" in str(raised.value)
    assert "#262" not in str(raised.value)


# --- the rows --------------------------------------------------------------------------------


@pytest.mark.parametrize("tranche", MINED)
def test_phase_b_takes_the_labelled_reported_rows_and_lists_the_unphonetizable(mined, tranche):
    decodable, listed = _select(mined, tranche)
    assert decodable == _decodable(tranche)
    assert listed == [row for row in _tranche(tranche) if row["row"] == 8]
    kept = {row["stratum"] for row in decodable}
    assert kept == {"reported:straddles", "reported:below_bar", "reported:identified",
                    "reported:unclean"}


def test_phase_b_never_takes_a_gating_or_an_unlabelled_unclean_row(mined):
    decodable, listed = _select(mined, "confirm.1")
    taken = _rows_of([*decodable, *listed])
    assert 1 not in taken and 2 not in taken and 7 not in taken


def test_phonetizability_is_asked_of_the_assigned_label_not_the_filename(mined):
    asked = []

    def phonetizable(label):
        asked.append(label)
        return _phonetizable(label)

    _select(mined, "confirm.1", phonetizable=phonetizable)
    assert sorted(set(asked)) == ["2:1", "2:2", "55:17"]


def test_a_filename_the_phonetizer_refuses_does_not_list_a_row_read_against_another_ayah(tmp_path):
    shard = SHARDS["confirm.1"]
    swapped = [_manifest_row(3, shard_file=shard, stratum="reported:identified",
                             label_source="identified", assigned_label="2:2",
                             filename_label="55:17"),
               _manifest_row(4, shard_file=shard, stratum="reported:identified",
                             label_source="identified", assigned_label="55:17",
                             filename_label="2:2")]
    fixture = _claimed(tmp_path, {"confirm.1": swapped, "confirm.2": _tranche("confirm.2")})
    decodable, listed = _select(fixture, "confirm.1")
    assert _rows_of(decodable) == [3] and _rows_of(listed) == [4]


@pytest.mark.parametrize("tranche", ["look", "confirm.3", "confirm.4", "train", "confirm", ""])
def test_no_other_tranche_is_selected(mined, tranche):
    with pytest.raises(ValueError):
        _select(mined, tranche)


def test_a_refused_tranche_opens_no_file(tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError):
        phase_b_rows("look", tranches_dir=empty, split_dir=empty, phonetizable=_phonetizable)


def test_phase_b_still_needs_the_committed_claim(tmp_path):
    fixture = _claimed(tmp_path, {role: _tranche(role) for role in MINED}, claims=())
    with pytest.raises(ValueError, match="claim"):
        _select(fixture, "confirm.1")


def test_a_mineable_row_the_split_tree_gives_to_train_is_refused(mined, tmp_path):
    round_, _ = mined
    victim = _decodable("confirm.1")[0]
    key = (victim["split"], victim["shard_file"], victim["row"])
    split = _split_tree(tmp_path / "train-split",
                        {role: _tranche(role) for role in MINED}, override={key: "train"})
    with pytest.raises(ValueError, match="train"):
        _select((round_, split), "confirm.1")


def test_the_real_confirm_manifests_give_ten_thousand_six_hundred_and_sixty_eight_rows():
    """The committed trees: 10,668 rows, 26 of them listed (#248), the issue's ≈ 10,600."""
    counts = {}
    for tranche in MINED:
        decodable, listed = phase_b_rows(tranche, phonetizable=_phonetizable)
        assert {row["assigned_label"] for row in listed} <= UNPHONETIZABLE_AYAT
        assert all(row["assigned_label"] is not None and row["stratum"] != "gating"
                   for row in [*decodable, *listed])
        counts[tranche] = (len(decodable), len(listed),
                           dict(Counter(row["label_source"] for row in decodable)))
    assert counts == {
        "confirm.1": (5325, 16, {"filename": 1261, "both": 1303, "identified": 2761}),
        "confirm.2": (5317, 10, {"filename": 1274, "both": 1331, "identified": 2712})}
    assert sum(decodable + listed for decodable, listed, _ in counts.values()) == 10_668


# --- the decode batch -------------------------------------------------------------------------


class Clock:
    """Time advances only when a decoder runs: one second per ``app_checks`` call."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class FakeDecoder:
    def __init__(self, name: str, clock: Clock):
        self.name, self.clock, self.calls = name, clock, 0

    def app_checks(self, span):
        self.calls += 1
        self.clock.now += 1.0
        return [{"hop": f"{self.name}:{len(span)}", "overlap": "", "flush": True}]

    def decode_spans(self, spans):
        return [f"{self.name}:spans:{len(span)}" for span in spans]


def _no_stream(name):
    raise AssertionError(f"every clip is cached, but {name} was streamed")


def _load_audio(data: bytes) -> np.ndarray:
    return np.zeros(len(data), dtype=np.float32)


def _batch_rows(tmp_path):
    """Four rows over two shards (two each), every clip in the clip cache."""
    clips = tmp_path / "clips"
    clips.mkdir()
    rows = []
    for shard, numbers in (("clean-00101-of-00411", (1, 2)), ("clean-00102-of-00411", (3, 4))):
        for number in numbers:
            payload = f"clip-{number}".encode()
            rows.append(_manifest_row(number, shard_file=shard, payload=payload,
                                      stratum="reported:below_bar", label_source="filename"))
            ted.clip_path(clips, rows[-1]["audio_sha256"]).write_bytes(payload)
    return rows, clips


def _decode(rows, clips, parts, decoders, clock, *, budget):
    return mined_decode.decode_batch(
        rows, parts, role="confirm.1", decoders=decoders, stream=_no_stream,
        word_counts=lambda label: 3, budget_seconds=budget, clock=clock,
        share_features=False, load_audio=_load_audio, clip_cache=clips)


def _decoders(clock):
    return {name: FakeDecoder(name, clock) for name in MODELS}


def _part(parts, shard, model):
    path = ted.decodes_path(parts, shard, model, role="confirm.1")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_mined_models_and_stores_are_the_round_two_set_in_checks_and_spans():
    assert mined_decode.MINED_MODELS == MODELS
    assert set(mined_decode.MINED_STORES) == {"checks", "spans"}
    assert mined_decode.MINED_STORES["checks"] is truth_baseline.CHECK_STORE
    assert dict(mined_decode.MINED_STORES["spans"].protocols) == {
        "spans": truth_baseline.PROTOCOLS["spans"]}


def test_a_batch_writes_every_model_a_record_with_both_stores_and_the_reference_source(tmp_path):
    rows, clips = _batch_rows(tmp_path)
    clock = Clock()
    result = _decode(rows, clips, tmp_path / "parts", _decoders(clock), clock, budget=1000)
    assert isinstance(result, BatchResult) and result.done
    for model in MODELS:
        lines = _part(tmp_path / "parts", "clean-00101-of-00411", model)
        assert [line["key"] for line in lines] == [item_key(row["audio_path"])
                                                   for row in rows[:2]]
        first = lines[0]
        assert first["checks"] == [{"hop": f"{model}:{len('clip-1')}", "overlap": "",
                                    "flush": True}]
        assert first["spans"] == f"{model}:spans:{len('clip-1')}"
        assert first["assigned_label"] == "2:1" and first["stratum"] == "reported:below_bar"


def test_the_budget_stops_between_rows_and_never_inside_one(tmp_path):
    rows, clips = _batch_rows(tmp_path)
    clock = Clock()
    result = _decode(rows, clips, tmp_path / "parts", _decoders(clock), clock, budget=1.0)
    assert not result.done
    held = {model: {line["key"] for shard in ("clean-00101-of-00411", "clean-00102-of-00411")
                    for line in _maybe_part(tmp_path / "parts", shard, model)}
            for model in MODELS}
    assert len(set(map(frozenset, held.values()))) == 1, "every model holds the same rows"
    assert 0 < len(held["base"]) < len(rows)


def _maybe_part(parts, shard, model):
    path = ted.decodes_path(parts, shard, model, role="confirm.1")
    return _part(parts, shard, model) if path.exists() else []


def test_a_zero_budget_decodes_nothing(tmp_path):
    rows, clips = _batch_rows(tmp_path)
    clock = Clock()
    decoders = _decoders(clock)
    result = _decode(rows, clips, tmp_path / "parts", decoders, clock, budget=0.0)
    assert not result.done and all(decoder.calls == 0 for decoder in decoders.values())


def test_a_resumed_batch_decodes_only_the_rows_the_parts_do_not_hold(tmp_path):
    rows, clips = _batch_rows(tmp_path)
    clock = Clock()
    decoders = _decoders(clock)
    assert not _decode(rows, clips, tmp_path / "parts", decoders, clock, budget=1.0).done
    assert _decode(rows, clips, tmp_path / "parts", decoders, clock, budget=1000).done
    assert {decoder.calls for decoder in decoders.values()} == {len(rows)}
    assert _decode(rows, clips, tmp_path / "parts", decoders, clock, budget=1000).done
    assert {decoder.calls for decoder in decoders.values()} == {len(rows)}


# --- the merge --------------------------------------------------------------------------------


SHARD_OF = {"confirm.1": "clean-00101-of-00411", "confirm.2": "clean-00102-of-00411"}


def _line(row: dict, model: str) -> dict:
    return {"key": item_key(row["audio_path"]), "audio_filename": row["audio_path"],
            "assigned_label": row["assigned_label"], "stratum": row["stratum"],
            "checks": [{"hop": model}], "spans": f"{model}:spans"}


def _write_parts(parts: Path, rows, *, models=MODELS, skip=(), extra=None) -> None:
    """``extra`` maps a tranche to additional rows written into its parts."""
    extra = extra or {}
    for tranche, tranche_rows in rows.items():
        for model in models:
            lines = [_line(row, model) for row in [*tranche_rows, *extra.get(tranche, ())]
                     if (tranche, model, row["row"]) not in skip]
            path = ted.decodes_path(parts, SHARD_OF[tranche], model, role=tranche)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(json.dumps(line) + "\n" for line in lines),
                            encoding="utf-8")
    for model in models:
        (parts / f"{model}.identity.json").write_text(
            json.dumps({"identity": {"model_ref": model}, "fingerprint": {"store": "checks"}}),
            encoding="utf-8")


def _merge_inputs(tmp_path):
    rows = {role: [{**row, "shard_file": SHARD_OF[role]} for row in _decodable(role)]
            for role in MINED}
    listed = {role: [{**row, "shard_file": SHARD_OF[role]}
                     for row in _tranche(role) if row["row"] == 8] for role in MINED}
    return rows, listed


def _merge(tmp_path, rows, listed, **kwargs):
    return merge_mined_decodes(tmp_path / "parts", tmp_path / "out", rows=rows,
                               unphonetizable=listed, provenance={"code_sha": "abc"},
                               models=MODELS, **kwargs)


def test_the_merge_pools_both_tranches_per_model_with_the_reference_source(tmp_path):
    rows, listed = _merge_inputs(tmp_path)
    _write_parts(tmp_path / "parts", rows)
    assert _merge(tmp_path, rows, listed) == {model: 8 for model in MODELS}
    for model in MODELS:
        lines = read_jsonl_gz(tmp_path / "out" / f"tlog.mined.{model}.jsonl.gz")
        assert [line["key"] for line in lines] == sorted(line["key"] for line in lines)
        by_key = {item_key(row["audio_path"]): (tranche, row)
                  for tranche in MINED for row in rows[tranche]}
        assert len(lines) == len(by_key)
        for line in lines:
            tranche, row = by_key[line["key"]]
            assert line["tranche"] == tranche
            assert line["label_source"] == row["label_source"]
            assert line["assigned_label"] == row["assigned_label"]
            assert line["spans"] == f"{model}:spans" and line["checks"] == [{"hop": model}]


def test_the_meta_reports_counts_per_label_source_the_listed_rows_and_the_identities(tmp_path):
    rows, listed = _merge_inputs(tmp_path)
    _write_parts(tmp_path / "parts", rows)
    _merge(tmp_path, rows, listed)
    meta = json.loads((tmp_path / "out" / "meta.json").read_text(encoding="utf-8"))
    assert meta["code_sha"] == "abc" and meta["models"] == list(MODELS)
    assert meta["rows"] == {model: 8 for model in MODELS}
    assert meta["label_sources"] == dict(Counter(
        row["label_source"] for tranche in MINED for row in rows[tranche]))
    assert sorted((entry["tranche"], entry["key"], entry["assigned_label"])
                  for entry in meta["unphonetizable"]) == sorted(
        (tranche, item_key(row["audio_path"]), "55:17") for tranche in MINED
        for row in listed[tranche])
    assert set(meta["identities"]) == {f"tlog.mined/{model}" for model in MODELS}
    assert meta["identities"]["tlog.mined/c95"]["identity"] == {"model_ref": "c95"}


@pytest.mark.parametrize("problem", ["missing", "repeated", "listed", "absent model"])
def test_a_partial_or_foreign_part_is_refused_and_nothing_is_written(tmp_path, problem):
    rows, listed = _merge_inputs(tmp_path)
    skip = {"missing": {("confirm.2", "c95", 5)}}.get(problem, ())
    extra = {"listed": {"confirm.1": listed["confirm.1"]},
             "repeated": {"confirm.1": rows["confirm.1"][:1]}}.get(problem)
    models = MODELS[:2] if problem == "absent model" else MODELS
    _write_parts(tmp_path / "parts", rows, models=models, skip=skip, extra=extra)
    with pytest.raises(ValueError):
        _merge(tmp_path, rows, listed)
    assert not (tmp_path / "out").exists() or not list((tmp_path / "out").iterdir())


def test_a_clip_in_both_tranches_is_refused(tmp_path):
    rows, listed = _merge_inputs(tmp_path)
    rows["confirm.2"][0] = {**rows["confirm.1"][0], "shard_file": SHARD_OF["confirm.2"]}
    _write_parts(tmp_path / "parts", rows)
    with pytest.raises(ValueError, match="both"):
        _merge(tmp_path, rows, listed)


# --- the driver -------------------------------------------------------------------------------


def _driver_args(tmp_path, phase: str):
    return SimpleNamespace(
        phase=phase, resume=True, parallel=2, parts_dir=tmp_path, clip_dir=tmp_path,
        shard_cache=tmp_path, tranches_dir=tmp_path, split_dir=tmp_path,
        pause_file=tmp_path / "PAUSE", lock=tmp_path / "gpu.lock", batch_seconds=300,
        yield_seconds=30, disk_reserve_gb=0.0, ram_floor_gb=0.0, look_clip_cache=None)


def _quiet_box(monkeypatch):
    monkeypatch.setattr(cli, "gpu_memory_mib", lambda field: 40_000)
    monkeypatch.setattr(cli, "_available_ram_gb", lambda: 64.0)
    monkeypatch.setattr(cli, "run_give_way",
                        lambda batch, **kwargs: GiveWayResult("done", 0, 0.0, 0.0))


def test_the_phase_b_driver_reads_the_mined_tranches_through_phase_b_selection(
        tmp_path, monkeypatch):
    _quiet_box(monkeypatch)
    asked = []

    def rows(tranche, **kwargs):
        asked.append(tranche)
        return [], []

    def never(*args, **kwargs):
        raise AssertionError("phase A's selection is not phase B's")

    monkeypatch.setattr(cli, "phase_b_rows", rows)
    monkeypatch.setattr(cli, "phase_a_rows", never)
    assert cli.run(_driver_args(tmp_path, "B")) == 0
    assert asked == list(MINED)


def test_the_driver_refuses_phase_c_before_it_touches_the_card(tmp_path, monkeypatch):
    def card(field):
        raise AssertionError("the card was read for a refused phase")

    monkeypatch.setattr(cli, "gpu_memory_mib", card)
    with pytest.raises(ValueError, match="phase C"):
        cli.run(_driver_args(tmp_path, "C"))


def test_the_driver_runs_one_phase_b_worker_whatever_the_card_offers(tmp_path, monkeypatch):
    """Phase B's worker peak is unmeasured: its cap is fixed, not read off a phase A figure."""
    _quiet_box(monkeypatch)
    monkeypatch.setattr(cli, "phase_b_rows", lambda role, **kwargs: ([], []))
    args = _driver_args(tmp_path, "B")
    assert cli.run(args) == 0
    assert args.parallel == 1
    assert cli._worker_count(4, 40_000, "B") == 1
    assert cli._worker_count(4, 40_000) == 4


# --- the real phonetizer, and the decode worker's wiring ---------------------------------------


def test_the_real_phonetizer_refuses_exactly_the_committed_fallback_ayat():
    """The fake above names #248's ayat from a literal; this is the code the box runs."""
    readable = selection._phonetizer()
    assert sorted(label for label in UNPHONETIZABLE_AYAT if not readable(label)) == sorted(
        UNPHONETIZABLE_AYAT)
    assert readable("2:1") and readable("1:1")
    from training.tlog_eval import KNOWN_UNPHONETIZABLE

    assert set(UNPHONETIZABLE_AYAT) == set(KNOWN_UNPHONETIZABLE)


def _decode_args(tmp_path, *, budget=60.0):
    return SimpleNamespace(
        phase="B", role="confirm.1", shard_index=0, shard_count=1,
        parts_dir=tmp_path / "parts", clip_dir=tmp_path / "clips",
        shard_cache=tmp_path / "shards", tranches_dir=tmp_path, split_dir=tmp_path,
        look_clip_cache=None, worklist=None, budget_seconds=budget)


def _listing(tmp_path) -> None:
    (tmp_path / "parts").mkdir(parents=True, exist_ok=True)
    (tmp_path / "parts" / "unphonetizable.confirm.1.json").write_text(
        json.dumps({"tranche": "confirm.1", "labels": []}), encoding="utf-8")


def test_the_decode_worker_refuses_before_the_models_without_the_launch_listing(tmp_path):
    with pytest.raises(ValueError, match="unphonetizable"):
        cli.decode(_decode_args(tmp_path))


def test_the_decode_worker_checks_the_producer_set_then_decodes_and_skips_held_shards(
        tmp_path, monkeypatch):
    rows, _ = _batch_rows(tmp_path)
    _listing(tmp_path)
    args = _decode_args(tmp_path)
    order, fetched, batches = [], [], []

    def confirm(models):
        order.append("confirm")

    def load(parts_dir, models=None, spec=None):
        order.append("load")
        return {}

    def batch(shard_rows, parts, *, role, **kwargs):
        batches.append([row["row"] for row in shard_rows])
        for model in MODELS:
            path = ted.decodes_path(parts, shard_rows[0]["shard_file"], model, role=role)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("".join(
                json.dumps({"key": item_key(row["audio_path"])}) + "\n"
                for row in shard_rows), encoding="utf-8")
        return BatchResult(done=True, written=len(shard_rows))

    monkeypatch.setattr(cli, "phase_b_rows", lambda role, **kwargs: (rows, []))
    monkeypatch.setattr(cli.ted, "require_confirm_models", confirm)
    monkeypatch.setattr(cli.ted, "load_models", load)
    monkeypatch.setattr(cli.ted, "ayah_word_counts", lambda: (lambda label: 3))
    monkeypatch.setattr(cli.ted, "shard_stream", lambda cache: _no_stream)
    monkeypatch.setattr(cli, "_fetch_shard",
                        lambda shard, shard_rows, a: fetched.append(shard))
    monkeypatch.setattr(cli, "decode_batch", batch)

    assert cli.decode(args) == 0
    assert order == ["confirm", "load"]
    assert fetched == ["clean-00101-of-00411", "clean-00102-of-00411"]
    assert batches == [[1, 2], [3, 4]]
    assert cli.decode(args) == 0
    assert fetched == ["clean-00101-of-00411", "clean-00102-of-00411"] and len(batches) == 2


def test_the_decode_worker_fetches_no_shard_once_its_budget_is_spent(tmp_path, monkeypatch):
    rows, _ = _batch_rows(tmp_path)
    _listing(tmp_path)
    args = _decode_args(tmp_path, budget=0.0)
    fetched, batches = [], []
    monkeypatch.setattr(cli, "phase_b_rows", lambda role, **kwargs: (rows, []))
    monkeypatch.setattr(cli.ted, "require_confirm_models", lambda models: None)
    monkeypatch.setattr(cli.ted, "load_models", lambda parts_dir, models=None: {})
    monkeypatch.setattr(cli.ted, "ayah_word_counts", lambda: (lambda label: 3))
    monkeypatch.setattr(cli.ted, "shard_stream", lambda cache: _no_stream)
    monkeypatch.setattr(cli, "_fetch_shard",
                        lambda shard, shard_rows, a: fetched.append(shard))
    monkeypatch.setattr(cli, "decode_batch",
                        lambda rows, parts, **kwargs: batches.append(rows))
    assert cli.decode(args) == 0
    assert fetched == [] and batches == []

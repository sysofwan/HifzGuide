"""Acceptance tests for tlog S6 identification and audit (``tadabur.tlog_ident_run``, #191).

Committed before the module exists and **locked**: later turns may add tests but never weaken or
rewrite these. The rules are the frozen ``docs/tlog-holdout-preregistration.md`` ("Ayah
identification", "Training pool" and "Audit (S6)") as amended by architect r20, whose decisions
are binding and are restated in the module contract below. The new dated amendment is
``docs/tlog-s6-amendment.md``; it must be committed, with ``decode_limit.json``, before the first
tlog decode.

Contract this file pins
-----------------------
``tadabur.tlog_ident_run`` exposes:

* Constants ``WHOLE_CLIP = "whole_clip"``, ``NOT_DECODED = "not_decoded:over_d_max"``,
  ``SAMPLE_SIZE = 2000``, ``FIVE_PERCENT_LIMIT = 0.05``, ``D_MAX_STOP_SHARE = 0.01``,
  ``SALT`` (the draw salt, written in the amendment), the frozen verdict vocabulary
  ``RIGHT = "right ayah"``, ``OTHER = "other ayah"``, ``SILENCE = "silence"``,
  ``NOT_RECITATION = "not recitation"``, ``UNCLEAR = "unclear"``, and the frozen audit sizes
  ``AUDIT_SIZES`` (``both`` 40, ``relabelled`` 30, ``non_recitation`` 25, ``silence`` 15,
  ``unclean identified`` 25, ``below-bar disagreement`` 15).
* Paths: ``DECODES_DIR = tools/tadabur/tlog/ident/decodes``,
  ``IDENT_DIR = tools/tadabur/tlog/ident``, ``AUDIT_DIR = tools/tadabur/tlog/audit``,
  ``DECODE_LIMIT_PATH = IDENT_DIR/decode_limit.json``,
  ``AMENDMENT_PATH = docs/tlog-s6-amendment.md``, ``DOC_PATH = docs/tlog-audit.md``.
  The committed trees are one JSONL.gz per population and split (r20 Q2):
  ``decodes/look/{clean,unclean}.jsonl.gz``, ``decodes/train_sample.jsonl.gz``,
  ``ident/look/{clean,unclean}.jsonl.gz``, ``ident/train_sample.jsonl.gz``, plus each tree's
  ``meta.json`` (``ident/train_sample.meta.json`` carries the 5% rule). Each file is sorted by
  ROW_KEY ``(split, shard_file, row)``; ``meta.json`` carries ``model``, ``revision``,
  ``code_sha``, ``calibration_sha256``, ``decode_limit_sha256`` and ``amendment_sha256``, and a
  per-shard census that lists zero-row shards.
* ``decode_limit(path=...) -> dict``: the committed ``decode_limit.json`` — ``d_max_seconds``
  (the largest ladder rung with ``headroom_gib >= 1.5``), ``gpu``, and ``rungs`` over the ladder
  60, 90, 120, 150, 180, 240, 300 s with ``peak_gib``/``headroom_gib``. A malformed record is a
  refusal, never a default.
* ``require_decode_preconditions()``: refuses unless the amendment and ``decode_limit.json``
  exist (they must be committed before the first tlog decode).
* ``select(d_max, *, count=2000, salt=None, rows=None) -> Selection`` with ``.look``,
  ``.train_sample``, ``.over_d_max`` (all tuples of row dicts in ROW_KEY order) and ``.counts``
  (``look_rows``, ``look_selected``, ``look_over_d_max``, ``train_rows``, ``train_selected``,
  ``train_over_d_max``). Look is every usable ``look`` row (9,710) with ``duration_s <= d_max``;
  the sample is the first 2,000 usable ``train`` rows with ``duration_s <= d_max`` by salted rank,
  re-derivable from ``salt`` and the row key. Rows above ``d_max`` are listed in ``.over_d_max``,
  carry ``decode_mode = NOT_DECODED`` and ``label_source = "none"``, and are never decoded.
  No ``confirm.*`` row is ever selected.
* ``decode_record(**key, decode_mode, spans)``: the committed decode row; ``decode_mode`` is a
  required keyword. ``identification_record(identification, *, decode_mode)`` is S5's row contract
  plus that required field, and ``not_decoded_record(row, *, d_max)`` is the same key set for a
  row above ``d_max`` with ``label_source = "none"``. S5's calibration keeps its own recorded
  modes and is re-derived unchanged.
* ``decode_shard(decoder, shard, wanted, *, d_max, stream, load_audio=None) -> list[dict]``:
  streams one shard, decodes only ``wanted`` rows after checking each ``audio_sha256``, records
  ``WHOLE_CLIP``, and raises before decoding any wanted row whose ``duration_s > d_max``.
* ``band_of(duration_s, d_max) -> str | None`` and ``band_report(rows, d_max)``: the three
  pre-registered bands ``0-20`` (<= 20 s), ``20-60`` (20–60 s) and ``60-d_max`` (60 s to
  ``d_max``), with null for a row above ``d_max``; the report counts rows and ``non_recitation``
  per band.
* ``audit_draw(records, *, sizes=None, salt=None) -> AuditDraw`` with ``.items`` and
  ``.strata`` (per stratum ``target``, ``eligible``, ``drawn``, ``shortfall``; freed slots go to
  ``both`` so the draw stays at 150). Strata definitions are r20 Q3: ``both`` = ``label_source
  == "both"``; ``relabelled`` = clean, filename-labelled, ``label_source == "identified"``;
  ``unclean identified`` = unclean and identified; ``non_recitation`` = decoded with
  ``floor_ok`` false; ``silence`` = a look row whose only exclusion is ``silence`` (not decoded;
  at least 8 unclean, the rest clean, then unclean again); ``below-bar disagreement`` =
  filename-labelled, ``floor_ok`` true, best class different from the filename class, not
  relabelled, recording ``delta`` or ``overlap`` as the reason. Rows with
  ``decode_mode = NOT_DECODED`` are in no stratum; no ``confirm.*`` row is ever drawn.
* ``verdict_outcome(item, verdict)``: the r20 Q4 mapping, returning ``label_under_test``,
  ``correct`` (``None`` for unclear), ``best_correct`` (below-bar only) and ``recitation``.
* ``audit_payload(draw, texts=...)``: the served payload; the stratum, filename label and
  assigned label are never sent, and candidates are anonymous ayah texts.
* ``five_percent_rule(rows)`` and ``render_audit()``: the Wilson 95% outcome over the 2,000
  identifications, and the generated ``docs/tlog-audit.md`` (the committed file must equal a
  fresh render).

Test command, from ``tools/``::

    python -m pytest -q tadabur/test_tlog_ident_run.py
"""

from __future__ import annotations

import hashlib
import importlib
import json
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

from tadabur import tlog_audit as audit
from tadabur import tlog_pool as pool
from tadabur.tlog_overlap import iter_pool_rows
from tadabur.tlog_pass1 import read_jsonl_gz
from training.counterfactual_eval import wilson_interval

ROOT = Path(__file__).resolve().parents[2]
FROZEN = ROOT / "docs" / "tlog-holdout-preregistration.md"
FROZEN_SHA256 = "204fe7f980488aff73402c56f1729f7f17f7c9471ea0664f6365bb8870b62086"
FLOOR = ROOT / "docs" / "tlog-ident-floor-preregistration.md"
FLOOR_SHA256 = "7bd2e1958ea8330df472030aefb6c97de7d0e9ccc2aa81ac133db682e1d58e3a"
NOTE = ROOT / "docs" / "tlog-s6-amendment-note-overlap.md"
NOTE_SHA256 = "e47856b8f34b1bd10704498a896fdf7fcc88d903705eb665d12763ed675e285a"
GATING = ROOT / "docs" / "tlog-gating-amendment.md"
GATING_SHA256 = "4fe70ed08e2d8fa1594eccb094986e131482882f309c5c4eb773c5685609f747"
AMENDMENT = ROOT / "docs" / "tlog-s6-amendment.md"
CALIBRATION = ROOT / "tools" / "tadabur" / "tlog" / "ident" / "ident_calibration.json"
LOOK_ROWS = 9_710
TRAIN_ROWS = 426_825
LOOK_OVER_60 = 28
TRAIN_OVER_60 = 1_359
LADDER = (60, 90, 120, 150, 180, 240, 300)
STRATA = ("both", "relabelled", "non_recitation", "silence", "unclean identified",
          "below-bar disagreement")
SIZES = {"both": 40, "relabelled": 30, "non_recitation": 25, "silence": 15,
         "unclean identified": 25, "below-bar disagreement": 15}
IDENTIFICATION_KEYS = frozenset({
    "surah", "ayah_from", "word_from", "ayah_to", "word_to", "straddles", "score", "length",
    "class_label", "decoy", "exhaustive_decoy", "delta", "margin", "margin_class", "id_floor_ok",
    "overlaps_filename", "filename_label", "label_source", "assigned_label", "floor_rule",
    "decode_mode"})
PROVENANCE = ("split", "shard_file", "row", "audio_path", "audio_sha256")


@pytest.fixture(scope="session")
def run():
    return importlib.import_module("tadabur.tlog_ident_run")


@pytest.fixture(scope="session")
def ident():
    return importlib.import_module("tadabur.tlog_ident")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _key(row: dict) -> tuple[str, str, int]:
    return row["split"], row["shard_file"], row["row"]


def _keys(rows) -> list[tuple[str, str, int]]:
    return [_key(row) for row in rows]


def _keyset(rows) -> set[tuple[str, str, int]]:
    return {_key(row) for row in rows}


@lru_cache(maxsize=1)
def _pool() -> tuple[dict, ...]:
    """The usable look and train rows with their frozen duration, once per worker."""
    rows = []
    for row in iter_pool_rows():
        if row["role"] not in ("look", "train") or not row["usable"]:
            continue
        rows.append({field: row[field] for field in (
            "split", "shard_file", "row", "role", "usable", "filename_label", "audio_path",
            "audio_sha256", "duration_s")})
    return tuple(rows)


def _fake_decoder() -> object:
    class _Decoder:
        def __init__(self) -> None:
            self.spans = 0

        def decode_spans(self, spans):
            spans = list(spans)
            self.spans += len(spans)
            return ["بجح" for _ in spans]

    return _Decoder()


def _stream(rows):
    return lambda shard: iter(rows)


def _candidate(run, *, population="look", split="clean", role="look", row=1, **overrides):
    """One audit candidate, with the fields the r20 Q3 definitions read."""
    candidate = {
        "population": population, "split": split, "role": role, "row": row,
        "shard_file": f"{split}-00000-of-00411",
        "audio_path": f"100_10_{row}.flac", "audio_sha256": _sha256(str(row).encode()),
        "filename_label": "2:1", "label_source": "both", "id_floor_ok": True,
        "class_label": "2:1", "assigned_label": "2:1", "delta": None,
        "overlaps_filename": False, "exclusions": (),
        "decode_mode": pool.WHOLE_CLIP,
    }
    candidate.update(overrides)
    return candidate


def _draw(run, candidates, sizes=None, salt="audit-test"):
    return audit.audit_draw(candidates, sizes=SIZES if sizes is None else sizes, salt=salt)


# --- the order and the pinned paths (r20 Q2/Q9) ------------------------------------------------


def test_the_amendment_and_decode_limit_are_committed_and_cite_the_frozen_rules(run):
    assert run.AMENDMENT_PATH == ROOT / "docs" / "tlog-s6-amendment.md"
    assert pool.DECODE_LIMIT_PATH == ROOT / "tools" / "tadabur" / "tlog" / "ident" / \
        "decode_limit.json"
    assert run.DECODES_DIR == ROOT / "tools" / "tadabur" / "tlog" / "ident" / "decodes"
    assert run.IDENT_DIR == ROOT / "tools" / "tadabur" / "tlog" / "ident"
    assert audit.AUDIT_DIR == ROOT / "tools" / "tadabur" / "tlog" / "audit"
    assert audit.DOC_PATH == ROOT / "docs" / "tlog-audit.md"

    text = run.AMENDMENT_PATH.read_text(encoding="utf-8")
    assert FROZEN.is_file() and FLOOR.is_file()
    assert FROZEN_SHA256[:8] in text and FLOOR_SHA256[:8] in text
    assert "D_max" in text
    assert run.SALT and run.SALT in text
    assert re.search(r"2026-\d\d-\d\d", text)


def test_the_overlap_note_is_frozen_and_cites_the_amendment(run):
    assert audit.OVERLAP_NOTE_PATH == NOTE
    text = NOTE.read_text(encoding="utf-8")
    assert _sha256(NOTE.read_bytes()) == NOTE_SHA256
    assert _sha256(run.AMENDMENT_PATH.read_bytes()) in text
    assert "2026-10-27" in text
    assert "unclear" in text and "right ayah" in text
    assert "whatever other stratum they also belong to" in text


def test_the_gating_amendment_is_the_owner_approved_post_look_rule():
    """The dated post-look amendment S7 and S8 cite: pinned by SHA-256, dated 2026-10-28,
    approved on #173, and citing the frozen rules it supersedes for gating."""
    text = GATING.read_text(encoding="utf-8")
    assert _sha256(GATING.read_bytes()) == GATING_SHA256
    assert "2026-10-28" in text and "post-look" in text and "owner-approved" in text
    assert "#issuecomment-6466464463" in text
    assert "da76351" in text
    assert FROZEN_SHA256 in text and FLOOR_SHA256 in text
    assert _sha256(AMENDMENT.read_bytes()) in text and NOTE_SHA256 in text
    assert "`label_source == both`" in text
    assert "obadx/muaalem-model-v3_2@01a1ef9f" in text
    assert "no `non_recitation` term" in text
    assert "2,853" in text and "3,059" in text and "208" in text


def test_the_amendment_and_decode_limit_must_exist_before_a_decode(run, tmp_path, monkeypatch):
    monkeypatch.setattr(run, "AMENDMENT_PATH", tmp_path / "absent-amendment.md")
    with pytest.raises((FileNotFoundError, ValueError)):
        run.require_decode_preconditions()
    monkeypatch.setattr(run, "AMENDMENT_PATH", ROOT / "docs" / "tlog-s6-amendment.md")
    monkeypatch.setattr(pool, "DECODE_LIMIT_PATH", tmp_path / "absent-limit.json")
    with pytest.raises((FileNotFoundError, ValueError)):
        run.require_decode_preconditions()


def test_the_committed_decode_limit_is_the_largest_rung_with_headroom(run):
    limit = pool.decode_limit()
    assert [rung["seconds"] for rung in limit["rungs"]] == list(LADDER)
    assert limit["d_max_seconds"] in LADDER
    assert isinstance(limit["gpu"], str) and limit["gpu"]
    chosen = next(r for r in limit["rungs"] if r["seconds"] == limit["d_max_seconds"])
    assert chosen["headroom_gib"] >= 1.5
    assert all(r["headroom_gib"] < 1.5 for r in limit["rungs"]
               if r["seconds"] > limit["d_max_seconds"])


def test_decode_limit_refuses_a_malformed_or_unmeasured_record(run, tmp_path):
    path = tmp_path / "decode_limit.json"
    path.write_text(json.dumps({"schema": "x", "d_max_seconds": 120, "gpu": "L4",
                                "rungs": [{"seconds": 60, "peak_gib": 1.0,
                                           "headroom_gib": 2.0}]}))
    with pytest.raises(ValueError):
        pool.decode_limit(path)
    path.write_text(json.dumps({"schema": "x", "gpu": "L4", "rungs": []}))
    with pytest.raises(ValueError):
        pool.decode_limit(path)


# --- selection (r20 Q1/Q5) ---------------------------------------------------------------------


def test_the_look_selection_is_the_usable_look_rows_up_to_d_max(run):
    rows = _pool()
    usable = [row for row in rows if row["role"] == "look"]
    assert len(usable) == LOOK_ROWS

    unlimited = run.select(d_max=10_000.0, rows=rows)
    assert _keys(unlimited.look) == sorted(_keys(usable))
    assert unlimited.over_d_max == ()
    assert unlimited.counts["look_rows"] == LOOK_ROWS
    assert unlimited.counts["look_selected"] == LOOK_ROWS
    assert unlimited.counts["look_over_d_max"] == 0

    capped = run.select(d_max=60.0, rows=rows)
    expected = [row for row in usable if row["duration_s"] <= 60.0]
    over = [row for row in usable if row["duration_s"] > 60.0]
    assert len(over) == LOOK_OVER_60
    assert _keys(capped.look) == sorted(_keys(expected))
    listed = [row for row in capped.over_d_max if row["role"] == "look"]
    assert _keys(listed) == sorted(_keys(over))
    assert capped.counts["look_over_d_max"] == LOOK_OVER_60
    assert all(row["decode_mode"] == pool.NOT_DECODED and row["label_source"] == "none"
               for row in listed)
    assert all(row["duration_s"] > 60.0 for row in listed)


def test_the_train_sample_is_two_thousand_seeded_rows_below_d_max(run):
    rows = _pool()
    sample = run.select(d_max=60.0, count=2_000, salt="sample-test", rows=rows)
    assert len(sample.train_sample) == 2_000
    assert sample.counts["train_rows"] == TRAIN_ROWS
    assert sample.counts["train_over_d_max"] == TRAIN_OVER_60
    eligible = {_key(row) for row in rows
                if row["role"] == "train" and row["duration_s"] <= 60.0}
    assert _keyset(sample.train_sample) <= eligible
    assert all(row["duration_s"] <= 60.0 for row in sample.train_sample)
    assert _keys(sample.train_sample) == sorted(_keys(sample.train_sample))

    again = run.select(d_max=60.0, count=2_000, salt="sample-test",
                       rows=list(reversed(rows)))
    assert _keys(again.train_sample) == _keys(sample.train_sample)
    other = run.select(d_max=60.0, count=2_000, salt="sample-other", rows=rows)
    assert _keyset(other.train_sample) != _keyset(sample.train_sample)


def test_no_confirm_role_is_selected_listed_or_drawn(run):
    selection = run.select(d_max=60.0, rows=_pool())
    for row in (*selection.look, *selection.train_sample, *selection.over_d_max):
        assert row["role"] in ("look", "train"), row["role"]


# --- decode records and the decode pass (r20 Q1, S5b I3.3) -------------------------------------


def test_a_decode_record_requires_its_mode(run):
    fields = {"split": "clean", "shard_file": "clean-00000-of-00411", "row": 7,
              "audio_path": "100_10_1.flac", "audio_sha256": "a" * 64, "duration_s": 5.0,
              "spans": "بجح"}
    record = pool.decode_record(**fields, decode_mode=pool.WHOLE_CLIP)
    assert record["decode_mode"] == "whole_clip"
    assert record["audio_sha256"] == "a" * 64
    assert pool.NOT_DECODED == "not_decoded:over_d_max"
    with pytest.raises(TypeError):
        pool.decode_record(**fields)


def test_an_identification_record_requires_its_mode(ident, run):
    def ayah(surah, ayah_number, words):
        offsets = [0]
        for word in words:
            offsets.append(offsets[-1] + len(word))
        return ident.AyahReference(surah, ayah_number, "".join(words), tuple(offsets))

    database = ident.ReferenceDatabase.build([
        ayah(1, 1, ["بجح", "دذر", "زسط"]),
        ayah(2, 1, ["طظع", "غفق", "كلم"]),
    ])
    decoded = ident.identify("طظعغفقكل", database=database, filename_ayah="1:1",
                             bars={"b": 0.01, "b_u": 0.01})
    record = ident.identification_record(decoded, decode_mode=pool.WHOLE_CLIP)
    assert set(record) == IDENTIFICATION_KEYS
    assert record["decode_mode"] == pool.WHOLE_CLIP
    with pytest.raises(TypeError):
        ident.identification_record(decoded)


def test_a_row_above_d_max_is_recorded_as_not_decoded_with_no_label(run, ident):
    def ayah(surah, ayah_number, words):
        offsets = [0]
        for word in words:
            offsets.append(offsets[-1] + len(word))
        return ident.AyahReference(surah, ayah_number, "".join(words), tuple(offsets))

    database = ident.ReferenceDatabase.build([ayah(1, 1, ["بجح", "دذر"])])
    decoded = ident.identify("بجحدذر", database=database)
    record = pool.not_decoded_record(
        {"split": "clean", "shard_file": "clean-00000-of-00411", "row": 1,
         "audio_path": "100_10_1.flac", "audio_sha256": "a" * 64, "duration_s": 130.0,
         "filename_label": "1:1"},
        d_max=120.0)
    assert set(record) == set(ident.identification_record(decoded, decode_mode=pool.WHOLE_CLIP))
    assert record["decode_mode"] == pool.NOT_DECODED
    assert record["label_source"] == "none"
    assert record["assigned_label"] is None


def test_decode_shard_decodes_only_the_selected_rows(run):
    decoder = _fake_decoder()
    payloads = {0: b"row-zero", 1: b"row-one", 2: b"row-two"}
    wanted = [
        {"split": "clean", "shard_file": "clean-00000-of-00411", "row": 0,
         "audio_path": "a.flac", "audio_sha256": _sha256(payloads[0]), "duration_s": 4.0},
        {"split": "clean", "shard_file": "clean-00000-of-00411", "row": 2,
         "audio_path": "c.flac", "audio_sha256": _sha256(payloads[2]), "duration_s": 10.0},
    ]
    stream_rows = [{"audio": {"path": f"{index}.flac", "bytes": payload}}
                   for index, payload in payloads.items()]
    records = pool.decode_shard(decoder, "clean-00000-of-00411", wanted, d_max=60.0,
                               stream=_stream(stream_rows),
                               load_audio=lambda data: np.zeros(1_600, dtype=np.float32))
    assert _keys(records) == _keys([wanted[0], wanted[1]])
    assert decoder.spans == 2
    assert all(record["decode_mode"] == pool.WHOLE_CLIP for record in records)


def test_decode_shard_refuses_a_row_above_d_max_before_decoding(run):
    decoder = _fake_decoder()
    payload = b"row-one"
    wanted = [{"split": "clean", "shard_file": "clean-00000-of-00411", "row": 1,
               "audio_path": "b.flac", "audio_sha256": _sha256(payload),
               "duration_s": 61.0}]
    stream_rows = [{"audio": {"path": "b.flac", "bytes": payload}}]
    with pytest.raises(ValueError):
        pool.decode_shard(decoder, "clean-00000-of-00411", wanted, d_max=60.0,
                         stream=_stream(stream_rows),
                         load_audio=lambda data: np.zeros(1_600, dtype=np.float32))
    assert decoder.spans == 0


def test_decode_shard_checks_the_audio_checksum_before_decoding(run):
    decoder = _fake_decoder()
    wanted = [{"split": "clean", "shard_file": "clean-00000-of-00411", "row": 0,
               "audio_path": "a.flac", "audio_sha256": "0" * 64, "duration_s": 5.0}]
    stream_rows = [{"audio": {"path": "a.flac", "bytes": b"other"}}]
    with pytest.raises(ValueError):
        pool.decode_shard(decoder, "clean-00000-of-00411", wanted, d_max=60.0,
                         stream=_stream(stream_rows),
                         load_audio=lambda data: np.zeros(1_600, dtype=np.float32))
    assert decoder.spans == 0


# --- the 5% rule and per-band reporting (r20 Q1/Q5) --------------------------------------------


def test_the_five_percent_rule_uses_the_wilson_upper_bound(run):
    def rows(non_recitation: int, total: int = 2_000):
        return ([{"id_floor_ok": False, "decode_mode": pool.WHOLE_CLIP}
                 for _ in range(non_recitation)]
                + [{"id_floor_ok": True, "decode_mode": pool.WHOLE_CLIP}
                   for _ in range(total - non_recitation)])

    quiet = run.five_percent_rule(rows(0))
    assert quiet["n"] == 2_000 and quiet["non_recitation"] == 0
    assert quiet["rate"] == 0.0 and quiet["triggered"] is False
    assert tuple(quiet["wilson"]) == wilson_interval(0, 2_000)
    assert quiet["limit"] == run.FIVE_PERCENT_LIMIT == 0.05

    below = run.five_percent_rule(rows(80))
    assert below["triggered"] is False
    assert below["wilson"][1] < 0.05
    above = run.five_percent_rule(rows(85))
    assert above["triggered"] is True
    assert above["wilson"][1] > 0.05


def test_results_are_reported_by_the_three_pre_registered_duration_bands(run):
    assert tuple(run.DURATION_BANDS) == ("0-20", "20-60", "60-d_max")
    assert run.band_of(20.0, 120.0) == "0-20"
    assert run.band_of(20.000_001, 120.0) == "20-60"
    assert run.band_of(60.0, 120.0) == "60-d_max"
    assert run.band_of(120.0, 120.0) == "60-d_max"
    assert run.band_of(120.000_001, 120.0) is None

    rows = [{"duration_s": seconds, "id_floor_ok": not flag}
            for seconds, flag in ((10.0, False), (20.0, True), (20.5, False), (59.0, True),
                                  (60.0, False), (120.0, True), (121.0, True))]
    report = run.band_report(rows, 120.0)
    assert report["0-20"]["rows"] == 2 and report["0-20"]["non_recitation"] == 1
    assert report["20-60"]["rows"] == 2 and report["20-60"]["non_recitation"] == 1
    assert report["60-d_max"]["rows"] == 2 and report["60-d_max"]["non_recitation"] == 1


# --- the audit draw (r20 Q3) --------------------------------------------------------------------


def test_a_row_that_meets_two_predicates_is_drawn_in_both_strata(run):
    """Architect r22: §4 sets no precedence, so each stratum draws from its own full set."""
    row = _candidate(run, split="clean", label_source="both", id_floor_ok=False)
    assert audit._matching_strata(row) == ("both", "non_recitation")
    draw = audit.audit_draw([row], sizes=dict.fromkeys(audit.AUDIT_SIZES, 1))
    assert len(draw.items) == 1
    (item,) = draw.items
    assert item["strata"] == ["both", "non_recitation"]
    assert item["stratum"] == "both"
    assert draw.strata["both"]["drawn"] == draw.strata["non_recitation"]["drawn"] == 1


def test_the_non_recitation_draw_is_the_first_twenty_five_over_the_whole_flag(run):
    rows = ([_candidate(run, row=1_000 + i, split="clean", label_source="both",
                        id_floor_ok=False) for i in range(20)]
            + [_candidate(run, row=2_000 + i, split="unclean", filename_label=None,
                          label_source="none", id_floor_ok=False,
                          class_label=None, assigned_label=None) for i in range(20)])
    sizes = {name: 0 for name in STRATA}
    sizes["non_recitation"] = 25
    draw = _draw(run, rows, sizes=sizes)
    expected = sorted(rows, key=lambda r: pool.rank("audit-test", "non_recitation", *_key(r)))
    assert draw.strata["non_recitation"]["eligible"] == 40
    assert _keyset(draw.items) == _keyset(expected[:25])
    assert all("non_recitation" in item["strata"] for item in draw.items)


def test_a_shared_row_is_served_once_and_scored_in_each_stratum(run):
    shared = _candidate(run, row=1, split="clean", label_source="both", id_floor_ok=False)
    passed = _candidate(run, row=2, split="clean", label_source="both", id_floor_ok=True)
    draw = audit.audit_draw([shared, passed], sizes=dict.fromkeys(audit.AUDIT_SIZES, 2))
    assert len(draw.items) == 2
    assert sum(1 for item in draw.items if item["row"] == 1) == 1
    assert len(audit.audit_payload(draw, texts={"2:1": "الْحَمْدُ"})["items"]) == 2

    views = audit._stratum_views(draw.items)
    verdicts = {audit._item_key(shared): {"verdict": audit.RIGHT},
                audit._item_key(passed): {"verdict": audit.RIGHT}}
    both = audit._precision(views["both"], verdicts)
    floor = audit._precision(views["non_recitation"], verdicts)
    assert (both["correct"], both["decided"]) == (2, 2)
    assert (floor["recitation"], floor["n"]) == (1, 1)


def test_an_identical_recitation_scores_the_class_of_the_label_under_test(run, ident):
    database = ident.ReferenceDatabase.build([
        ident.AyahReference(2, 1, "بجح", (0, 3)),
        ident.AyahReference(2, 2, "بجح", (0, 3)),
    ])
    canonical = lambda label: ident.canonical_label(database, label)
    both = {"stratum": "both", "filename_label": "2:1", "assigned_label": "2:1",
            "class_label": "2:1"}
    assert audit.verdict_outcome(both, {"verdict": audit.OTHER, "ayah": "2:2"},
                               canonical=canonical)["correct"] is True
    assert audit.verdict_outcome(both, {"verdict": audit.OTHER, "ayah": "2:2"})["correct"] is False
    outcome = audit.verdict_outcome({"stratum": "relabelled", "filename_label": "2:1",
                                   "assigned_label": "2:2", "class_label": "2:2"},
                                  {"verdict": audit.UNCLEAR})
    assert (outcome["label_under_test"], outcome["correct"]) == ("2:2", None)


def test_the_frozen_audit_sizes_are_the_pre_registered_table(run):
    assert audit.AUDIT_SIZES == SIZES
    assert sum(audit.AUDIT_SIZES.values()) == 150
    assert (audit.RIGHT, audit.OTHER, audit.SILENCE, audit.NOT_RECITATION, audit.UNCLEAR) == (
        "right ayah", "other ayah", "silence", "not recitation", "unclear")


def test_each_stratum_is_defined_by_the_r20_rule(run):
    cases = {
        "both": _candidate(run, label_source="both"),
        "relabelled": _candidate(run, split="clean", label_source="identified",
                                 assigned_label="2:2", class_label="2:2"),
        "non_recitation": _candidate(run, split="unclean", filename_label=None,
                                     label_source="none", id_floor_ok=False,
                                     class_label=None, assigned_label=None),
        "unclean identified": _candidate(run, split="unclean", filename_label=None,
                                         label_source="identified", assigned_label="2:2",
                                         class_label="2:2"),
        "below-bar disagreement": _candidate(run, label_source="filename", class_label="2:2",
                                             delta=0.0),
    }
    for stratum, candidate in cases.items():
        sizes = {name: 0 for name in STRATA}
        sizes[stratum] = 1
        draw = _draw(run, [candidate], sizes=sizes)
        assert [item["stratum"] for item in draw.items] == [stratum]

    silence = _candidate(run, split="unclean", filename_label=None, label_source="none",
                         id_floor_ok=None, class_label=None, assigned_label=None,
                         exclusions=("silence",), decode_mode=None)
    sizes = {name: 0 for name in STRATA}
    sizes["silence"] = 1
    assert [item["stratum"] for item in _draw(run, [silence], sizes=sizes).items] == ["silence"]

    # Not one definition: a silence row with another exclusion, or a row above D_max.
    mixed = dict(silence, exclusions=("silence", "duplicate"))
    over = _candidate(run, split="unclean", filename_label=None, label_source="none",
                      id_floor_ok=False, class_label=None, assigned_label=None,
                      decode_mode=pool.NOT_DECODED)
    sizes = {name: 1 for name in STRATA}
    draw = _draw(run, [mixed, over], sizes=sizes)
    assert draw.items == ()


def test_a_short_stratum_takes_all_its_rows_and_frees_the_rest_to_both(run):
    candidates = (
        [_candidate(run, row=1_000 + i, split="clean", label_source="identified",
                    assigned_label="2:2", class_label="2:2") for i in range(2)]
        + [_candidate(run, row=1_100 + i, label_source="filename", class_label="2:2",
                      delta=0.0) for i in range(2)]
        + [_candidate(run, row=2_000 + i) for i in range(80)]
        + [_candidate(run, row=3_000 + i, split="unclean", filename_label=None,
                      label_source="none", id_floor_ok=False, class_label=None,
                      assigned_label=None) for i in range(30)]
        + [_candidate(run, row=4_000 + i, split="clean", filename_label=None,
                      label_source="none", id_floor_ok=None, class_label=None,
                      assigned_label=None, exclusions=("silence",), decode_mode=None)
           for i in range(10)]
        + [_candidate(run, row=4_100 + i, split="unclean", filename_label=None,
                      label_source="none", id_floor_ok=None, class_label=None,
                      assigned_label=None, exclusions=("silence",), decode_mode=None)
           for i in range(10)]
        + [_candidate(run, row=5_000 + i, split="unclean", filename_label=None,
                      label_source="identified", assigned_label="2:2", class_label="2:2")
           for i in range(30)]
        + [_candidate(run, row=6_000 + i, label_source="filename", class_label="2:2",
                      delta=0.0) for i in range(13)]
    )
    draw = _draw(run, candidates)
    census = draw.strata
    assert census["relabelled"] == {"target": 30, "eligible": 2, "drawn": 2, "shortfall": 28}
    assert census["both"] == {"target": 40, "eligible": 80, "drawn": 68, "shortfall": 0}
    assert census["non_recitation"]["drawn"] == 25
    assert census["silence"]["drawn"] == 15
    assert census["unclean identified"]["drawn"] == 25
    assert census["below-bar disagreement"]["drawn"] == 15
    assert len(draw.items) == 150
    assert sum(item["stratum"] == "both" for item in draw.items) == 68

    small = _draw(run, [_candidate(run, row=7_000 + i, split="unclean", filename_label=None,
                                   label_source="none", id_floor_ok=False, class_label=None,
                                   assigned_label=None) for i in range(5)]
                  + [_candidate(run, row=8_000 + i) for i in range(60)])
    assert small.strata["non_recitation"]["drawn"] == 5
    assert small.strata["non_recitation"]["shortfall"] == 20
    assert small.strata["both"]["drawn"] == 60


def test_the_silence_stratum_takes_at_least_eight_unclean_rows(run):
    def silence(row, split):
        return _candidate(run, row=row, split=split, filename_label=None,
                          label_source="none", id_floor_ok=None, class_label=None,
                          assigned_label=None, exclusions=("silence",), decode_mode=None)

    clean = [silence(1_000 + i, "clean") for i in range(10)]
    unclean = [silence(2_000 + i, "unclean") for i in range(10)]
    draw = _draw(run, clean + unclean)
    drawn = [item for item in draw.items if item["stratum"] == "silence"]
    assert len(drawn) == 15
    assert sum(item["split"] == "unclean" for item in drawn) == 8
    assert sum(item["split"] == "clean" for item in drawn) == 7

    short_clean = [silence(3_000 + i, "clean") for i in range(3)]
    many_unclean = [silence(4_000 + i, "unclean") for i in range(20)]
    draw = _draw(run, short_clean + many_unclean)
    drawn = [item for item in draw.items if item["stratum"] == "silence"]
    assert len(drawn) == 15
    assert sum(item["split"] == "unclean" for item in drawn) == 12
    assert sum(item["split"] == "clean" for item in drawn) == 3


def test_the_draw_is_salted_order_independent_and_carries_provenance(run):
    candidates = [_candidate(run, row=9_000 + i) for i in range(100)]
    sizes = {name: 0 for name in STRATA}
    sizes["both"] = 40
    first = _draw(run, candidates, sizes=sizes, salt="draw-test")
    reversed_ = _draw(run, list(reversed(candidates)), sizes=sizes, salt="draw-test")
    other = _draw(run, candidates, sizes=sizes, salt="draw-other")
    assert _keys(first.items) == _keys(reversed_.items)
    assert _keyset(first.items) != _keyset(other.items)
    assert len(first.items) == 40
    for item in first.items:
        for field in PROVENANCE:
            assert item[field] is not None or field == "audio_sha256"
        assert item["population"] in ("look", "train_sample")
        assert item["role"] in ("look", "train")


# --- the verdicts and the served payload (r20 Q4) ----------------------------------------------


def test_the_verdict_mapping_scores_the_label_under_test(run):
    both = {"stratum": "both", "filename_label": "2:1", "assigned_label": "2:1",
            "class_label": "2:1"}
    relabelled = {"stratum": "relabelled", "filename_label": "2:1", "assigned_label": "2:2",
                  "class_label": "2:2"}
    below = {"stratum": "below-bar disagreement", "filename_label": "2:1",
             "assigned_label": "2:1", "class_label": "2:2"}
    non_recitation = {"stratum": "non_recitation", "filename_label": None,
                      "assigned_label": None, "class_label": None}
    silence = {"stratum": "silence", "filename_label": None, "assigned_label": None,
               "class_label": None}
    right = {"verdict": audit.RIGHT}
    other = lambda ayah: {"verdict": audit.OTHER, "ayah": ayah}
    not_recitation = {"verdict": audit.NOT_RECITATION}
    unclear = {"verdict": audit.UNCLEAR}

    outcome = audit.verdict_outcome(both, right)
    assert (outcome["label_under_test"], outcome["correct"]) == ("2:1", True)
    assert audit.verdict_outcome(relabelled, right)["label_under_test"] == "2:2"
    assert audit.verdict_outcome({"stratum": "unclean identified", "filename_label": None,
                                "assigned_label": "2:2", "class_label": "2:2"},
                               right)["label_under_test"] == "2:2"

    outcome = audit.verdict_outcome(below, right)
    assert (outcome["label_under_test"], outcome["correct"], outcome["best_correct"]) == (
        "2:1", True, False)
    outcome = audit.verdict_outcome(below, other("2:2"))
    assert (outcome["correct"], outcome["best_correct"]) == (False, True)
    outcome = audit.verdict_outcome(below, other("3:1"))
    assert (outcome["correct"], outcome["best_correct"]) == (False, False)

    outcome = audit.verdict_outcome(non_recitation, not_recitation)
    assert (outcome["label_under_test"], outcome["correct"], outcome["recitation"]) == (
        None, True, False)
    outcome = audit.verdict_outcome(non_recitation, right)
    assert (outcome["correct"], outcome["recitation"]) == (False, True)
    assert audit.verdict_outcome(non_recitation, unclear)["correct"] is None
    assert audit.verdict_outcome(silence, {"verdict": audit.SILENCE})["correct"] is True
    outcome = audit.verdict_outcome(silence, right)
    assert (outcome["correct"], outcome["recitation"]) == (False, True)


def test_the_served_payload_hides_the_stratum_and_the_attribution(run):
    candidates = [
        _candidate(run, row=9_000, label_source="filename", class_label="2:2", delta=0.0),
        _candidate(run, row=9_001, label_source="both"),
    ]
    draw = _draw(run, candidates)
    payload = audit.audit_payload(draw, texts={"2:1": "الْحَمْدُ", "2:2": "رَبِّ"})
    assert payload["progress"] == {"answered": 0, "total": len(draw.items)}
    for item in payload["items"]:
        assert "stratum" not in item
        assert not ({"filename_label", "assigned_label", "label_source", "population"}
                    & set(item))
        assert item["candidates"]
        for candidate in item["candidates"]:
            assert candidate["text"]
            assert not ({"under_test", "correct", "label_source"} & set(candidate))
    below = next(item for item in payload["items"] if len(item["candidates"]) == 2)
    assert {candidate["ayah"] for candidate in below["candidates"]} == {"2:1", "2:2"}


# --- the generated report (brief criteria) -----------------------------------------------------


def test_a_repeated_ayah_is_not_a_below_bar_disagreement(run, ident):
    """Review R3.1: one text class is never a disagreement, whatever its label strings."""
    database = ident.load_references()
    assert ident.canonical_label(database, "26:190") == ident.canonical_label(database, "26:8")
    repeated = _candidate(run, row=1, filename_label="26:190", class_label="26:8",
                          label_source="both", id_floor_ok=True, delta=0.0)
    assert "below-bar disagreement" not in audit._matching_strata(repeated)
    genuine = _candidate(run, row=2, filename_label="37:28", class_label="37:27",
                         label_source="filename", id_floor_ok=True, delta=0.0)
    assert "below-bar disagreement" in audit._matching_strata(genuine)


def test_the_below_bar_frame_holds_only_class_disagreements(run, ident):
    database = ident.load_references()
    frame = [row for row in audit.audit_candidates()
             if "below-bar disagreement" in audit._matching_strata(row)]
    assert len(frame) == 756
    equal = [row for row in frame
             if ident.canonical_label(database, row["class_label"])
             == ident.canonical_label(database, row["filename_label"])]
    assert equal == []


def test_the_committed_worklist_marks_the_frozen_floor_sample(run):
    prefix = {audit._item_key(row) for row in audit._floor_prefix(run.IDENT_DIR)}
    worklist = audit.read_jsonl(audit.AUDIT_DIR / "worklist.jsonl")
    assert len(prefix) == audit.AUDIT_SIZES["non_recitation"] == 25
    assert {audit._item_key(row) for row in worklist if row["floor_sample"]} == prefix


def test_an_unclear_below_bar_answer_is_not_a_best_class_miss(run):
    """Review R2.2: the best-class readout divides by decided answers, never by answered ones."""
    rows = [{"stratum": "below-bar disagreement", "split": "clean",
             "shard_file": "clean-00001-of-00411", "row": number, "filename_label": "2:1",
             "class_label": "2:2"} for number in range(15)]
    verdicts = {audit._item_key(row): {"verdict": audit.OTHER, "ayah": "2:2"}
                for row in rows[:14]}
    verdicts[audit._item_key(rows[14])] = {"verdict": audit.UNCLEAR}
    stats = audit._precision(rows, verdicts)
    assert (stats["n"], stats["decided"], stats["best_correct"]) == (15, 14, 14)
    line = audit._best_class_line(stats)
    assert "**14 of 14 decided**" in line
    assert "(15 answered" in line
    assert "78.47%–100.00%" in line


def test_the_committed_audit_doc_is_a_fresh_render(run):
    assert audit.DOC_PATH.read_text(encoding="utf-8") == audit.render_audit()
    assert audit.render_audit() == audit.render_audit()


def test_the_committed_train_sample_meta_records_provenance_and_the_five_percent_rule(run):
    meta = json.loads((run.IDENT_DIR / "train_sample.meta.json").read_text(encoding="utf-8"))
    assert meta["model"] == "obadx/muaalem-model-v3_2"
    assert meta["revision"] == "01a1ef9fbe40d144ef845101e89ff924aed3fef5"
    assert re.fullmatch(r"[0-9a-f]{40}", meta["code_sha"])
    assert meta["calibration_sha256"] == _sha256(CALIBRATION.read_bytes())
    assert meta["decode_limit_sha256"] == _sha256(pool.DECODE_LIMIT_PATH.read_bytes())
    assert meta["amendment_sha256"] == _sha256(run.AMENDMENT_PATH.read_bytes())

    selection = meta["selection"]
    assert selection["salt"] == run.SALT
    assert selection["train_rows"] == TRAIN_ROWS
    assert selection["train_selected"] == 2_000
    limit = pool.decode_limit()["d_max_seconds"]
    assert selection["train_over_d_max"] == sum(
        1 for row in _pool() if row["role"] == "train" and row["duration_s"] > limit)
    rule = meta["five_percent_rule"]
    assert rule["n"] == 2_000
    assert rule["limit"] == 0.05
    assert isinstance(rule["triggered"], bool)


def test_the_non_recitation_rule_reads_the_floor_verdict(run):
    """Review I4.3: one public definition, and a row above D_max is neither flagged nor assigned."""
    look = pool.read_ident_tree("look", ident_dir=run.IDENT_DIR)
    assert len(look) == LOOK_ROWS
    assert len(pool.read_ident_tree("train_sample", ident_dir=run.IDENT_DIR)) == 2_000
    undecoded = [row for row in look if row["decode_mode"] == pool.NOT_DECODED]
    assert len(undecoded) == 7
    for row in undecoded:
        assert row["id_floor_ok"] is None
        assert pool.is_non_recitation(row) is False
        assert row["label_source"] == "none" and row["assigned_label"] is None
    assert pool.is_non_recitation({"id_floor_ok": False}) is True
    assert pool.is_non_recitation({"id_floor_ok": True}) is False
    assert "non_recitation" not in look[0]


def test_the_committed_trees_are_one_file_per_population_and_split(run):
    for split in ("clean", "unclean"):
        assert (run.DECODES_DIR / "look" / f"{split}.jsonl.gz").is_file()
        assert (run.IDENT_DIR / "look" / f"{split}.jsonl.gz").is_file()
    assert (run.DECODES_DIR / "train_sample.jsonl.gz").is_file()
    assert (run.IDENT_DIR / "train_sample.jsonl.gz").is_file()
    assert (audit.AUDIT_DIR / "worklist.jsonl").is_file()
    assert (audit.AUDIT_DIR / "verdicts.jsonl").is_file()


def test_the_look_decodes_and_identifications_cover_the_selected_population(run):
    selection = run.select(d_max=pool.decode_limit()["d_max_seconds"])
    expected = _keyset(selection.look)
    assert len(expected) == LOOK_ROWS - selection.counts["look_over_d_max"]
    decoded: set[tuple[str, str, int]] = set()
    identified: set[tuple[str, str, int]] = set()
    for split in ("clean", "unclean"):
        decoded |= _keyset(read_jsonl_gz(run.DECODES_DIR / "look" / f"{split}.jsonl.gz"))
        identified |= _keyset(read_jsonl_gz(run.IDENT_DIR / "look" / f"{split}.jsonl.gz"))
    assert decoded == expected
    assert identified == expected | _keyset(
        [row for row in selection.over_d_max if row["role"] == "look"])


def test_no_confirm_row_is_in_any_committed_tree(run):
    trees = [run.DECODES_DIR / "train_sample.jsonl.gz",
             run.IDENT_DIR / "train_sample.jsonl.gz",
             *(run.DECODES_DIR / "look" / f"{split}.jsonl.gz" for split in ("clean", "unclean")),
             *(run.IDENT_DIR / "look" / f"{split}.jsonl.gz" for split in ("clean", "unclean"))]
    for path in trees:
        for row in read_jsonl_gz(path):
            assert row["split"] in ("clean", "unclean")
    for row in audit.read_jsonl(audit.AUDIT_DIR / "worklist.jsonl"):
        assert not str(row["role"]).startswith("confirm")
        assert row["split"] in ("clean", "unclean")


def test_each_decode_meta_census_counts_its_own_tree(run):
    pairs = [
        (run.DECODES_DIR / "look" / "meta.json",
         read_jsonl_gz(run.DECODES_DIR / "look" / "clean.jsonl.gz")
         + read_jsonl_gz(run.DECODES_DIR / "look" / "unclean.jsonl.gz")),
        (run.DECODES_DIR / "train_sample.meta.json",
         read_jsonl_gz(run.DECODES_DIR / "train_sample.jsonl.gz")),
    ]
    for meta_path, rows in pairs:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        tree_rows = len(rows) if isinstance(meta["rows"], int) else sum(meta["rows"].values())
        assert sum(meta["census"].values()) == tree_rows
        assert tree_rows == len(rows)


def test_the_committed_sample_trees_are_the_salted_draw(run):
    selection = run.select(d_max=pool.decode_limit()["d_max_seconds"])
    expected = _keyset(selection.train_sample)
    assert len(expected) == 2_000
    for path in (run.DECODES_DIR / "train_sample.jsonl.gz",
                 run.IDENT_DIR / "train_sample.jsonl.gz"):
        assert _keyset(read_jsonl_gz(path)) == expected


def test_the_committed_worklist_is_the_draw_it_was_staged_from(run):
    worklist = audit.read_jsonl(audit.AUDIT_DIR / "worklist.jsonl")
    draw = audit.audit_draw(audit.audit_candidates())
    assert len(worklist) == len(draw.items)
    assert len({_key(row) for row in worklist}) == len(worklist)
    assert {_key(row): row["strata"] for row in worklist} == {
        _key(item): item["strata"] for item in draw.items}
    assert sum(census["drawn"] for census in draw.strata.values()) == 150

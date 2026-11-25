"""Word snips (#264): the sidecar that tells the listening UI which part of a clip to play.

The interface these tests fix, in ``tadabur.word_snips``:

* ``MIN_LL_PER_FRAME``: the reliability gate, a module constant (-3.5, owner-approved 2026-11-25). An aligned
  record whose ``ll_per_frame`` is **below** it is treated exactly as ``aligned: False``.
* ``ALIGNMENT`` / ``WHOLE_CLIP``: the two ``snip_source`` values (``"alignment"``, ``"whole_clip"``).
* ``Snip(site_id, audio_sha256, start_sample, end_sample, snip_source)``: a frozen dataclass.
  A whole-clip snip has ``start_sample`` and ``end_sample`` ``None``.
* ``build_snips(rows, records, *, min_ll_per_frame=MIN_LL_PER_FRAME) -> list[Snip]``: ``rows`` are
  worklist rows as raw dicts (``id, audio_sha256, surah_ayah, word_index, start_sample,
  end_sample``; ``end_sample`` is the decoded 16 kHz clip's length, the start is 0); ``records``
  are word-time records (``tadabur.tlog_mining.word_times``). One snip per row, sorted by id.
  The span is :func:`tadabur.listening_session.excerpt_span`'s rule (the word and one either side,
  padded by ``EXCERPT_PAD_S``, at least ``MIN_EXCERPT_S``, clamped to the clip) over the record's
  word times; the clip plays whole (``WHOLE_CLIP``) when the record is missing, is ``aligned:
  False``, is below the gate, names another ayah than the row, has no such word, or the rule
  itself falls back.
* ``write_snips(snips, path)`` / ``read_snips(path) -> dict[str, Snip]``: a JSONL file, one row per
  id, ``{id, audio_sha256, start_sample, end_sample, snip_source}``, sorted by id.
* ``main(argv)``: ``build --worklist W --word-times F [F ...] --out OUT [--report R]
  [--min-ll-per-frame X]``. It prints the fallback rate and, with ``--report``, writes JSON
  ``{rows, alignment, whole_clip, fallback_rate, reasons}`` (``reasons`` counts each whole-clip
  row once, so its values sum to ``whole_clip``).
"""

from __future__ import annotations

import gzip
import json

import pytest

from tadabur.listening_session import (
    EXCERPT_PAD_S,
    MIN_EXCERPT_S,
    excerpt_span,
)
from tadabur.mining_pool import PoolSegment
from tadabur.word_snips import (
    ALIGNMENT,
    MIN_LL_PER_FRAME,
    WHOLE_CLIP,
    Snip,
    build_snips,
    main,
    read_snips,
    write_snips,
)

RATE = 16000
CLIP = 12 * RATE
#: Eight words a second apart, each 0.8 s long: the shared fixture the rule is checked on.
PAIRS = [(1.0 + k, 1.8 + k) for k in range(8)]
#: Words so short that no excerpt reaches MIN_EXCERPT_S: the rule falls back.
SHORT_PAIRS = [(0.1 * k, 0.1 * k + 0.05) for k in range(8)]


def _expected(pairs, word_index, clip_samples=CLIP):
    """What ``excerpt_span`` gives for the same word times, as the rule's own caller shapes them:
    each word's onset plus the last word's end, over a segment that is the whole ayah."""
    boundaries = (*[start for start, _ in pairs], pairs[-1][1])
    segment = PoolSegment(0, 0, len(pairs), 0, clip_samples, "", tuple(range(len(pairs) + 1)),
                          True)
    return excerpt_span(segment, boundaries, 0, word_index, clip_samples)


def _sha(name: str) -> str:
    return f"{name:0<64}"[:64]


def _row(sha_name: str, word_index: int, *, ayah="1:1", clip_samples=CLIP, site=None) -> dict:
    return {"id": site or f"{sha_name}-{word_index}", "split": "clean", "shard_file": "clean-0",
            "row": 0, "audio_filename": f"{sha_name}.flac", "audio_sha256": _sha(sha_name),
            "start_sample": 0, "end_sample": clip_samples, "surah_ayah": ayah,
            "word_index": word_index, "word": "w"}


def _record(sha_name: str, pairs=PAIRS, *, ayah="1:1", ll=-0.2, aligned=True) -> dict:
    return {"key": f"{sha_name}.flac#0", "audio_sha256": _sha(sha_name), "surah_ayah": ayah,
            "aligned": aligned, "reason": None if aligned else "too few frames",
            "word_times": [list(pair) for pair in pairs] if aligned else None,
            "seconds_per_frame": 0.04 if aligned else None,
            "ll_per_frame": ll if aligned else None}


def _one(row, record, **kwargs) -> Snip:
    (snip,) = build_snips([row], [] if record is None else [record], **kwargs)
    return snip


def _span(snip: Snip) -> tuple[int, int]:
    assert snip.snip_source == ALIGNMENT
    return snip.start_sample, snip.end_sample


# --- the rule is excerpt_span's -------------------------------------------------------------


def test_the_excerpt_is_the_word_and_one_either_side_padded_as_the_hand_computed_case():
    # Word 3's neighbours are words 2 and 4: onset 3.0 s (word 2's start), offset 6.0 s (word 5's
    # start, the boundary after word 4), each padded by EXCERPT_PAD_S.
    snip = _one(_row("a", 3), _record("a"))

    assert _span(snip) == (round(3.0 * RATE - EXCERPT_PAD_S * RATE),
                           round(6.0 * RATE + EXCERPT_PAD_S * RATE))


@pytest.mark.parametrize("word_index", range(8))
def test_every_word_of_the_shared_fixture_gets_the_span_excerpt_span_gives(word_index):
    expected = _expected(PAIRS, word_index)
    assert expected != (0, CLIP)  # the fixture exercises the rule, not its fallback

    assert _span(_one(_row("a", word_index), _record("a"))) == expected


def test_the_excerpt_is_clamped_only_to_the_clip():
    clip = 100_000  # shorter than word 3's padded end (100,800)

    snip = _one(_row("a", 3, clip_samples=clip), _record("a"))

    assert _span(snip) == _expected(PAIRS, 3, clip) == (43200, clip)


def test_an_excerpt_shorter_than_the_minimum_plays_the_whole_clip():
    assert _expected(SHORT_PAIRS, 3) == (0, CLIP)  # excerpt_span's own fallback
    assert 0.3 + 2 * EXCERPT_PAD_S < MIN_EXCERPT_S

    snip = _one(_row("a", 3), _record("a", SHORT_PAIRS))

    assert (snip.snip_source, snip.start_sample, snip.end_sample) == (WHOLE_CLIP, None, None)


# --- what makes a clip play whole -----------------------------------------------------------


@pytest.mark.parametrize("why, row, record", [
    ("not aligned", _row("a", 3), _record("a", aligned=False)),
    ("no record for the clip", _row("a", 3), None),
    ("a record for another clip", _row("a", 3), _record("b")),
    ("the record aligns another ayah", _row("a", 3, ayah="1:2"), _record("a", ayah="1:1")),
    ("the word is past the record's words", _row("a", 8), _record("a")),
    ("ll_per_frame below the gate", _row("a", 3), _record("a", ll=MIN_LL_PER_FRAME - 0.01)),
])
def test_words_that_cannot_be_placed_play_the_whole_clip(why, row, record):
    snip = _one(row, record)

    assert (snip.snip_source, snip.start_sample, snip.end_sample) == (WHOLE_CLIP, None, None), why
    assert snip.site_id == row["id"]
    assert snip.audio_sha256 == row["audio_sha256"]


def test_a_likelihood_at_the_gate_is_still_placed_and_the_gate_is_a_parameter():
    at_gate = _record("a", ll=MIN_LL_PER_FRAME)
    assert _one(_row("a", 3), at_gate).snip_source == ALIGNMENT

    good = _record("a", ll=MIN_LL_PER_FRAME + 0.5)
    assert _one(_row("a", 3), good).snip_source == ALIGNMENT
    stricter = _one(_row("a", 3), good, min_ll_per_frame=MIN_LL_PER_FRAME + 1.0)
    assert stricter.snip_source == WHOLE_CLIP


def test_two_words_of_one_clip_each_get_their_own_snip_in_id_order():
    rows = [_row("a", 5, site="z"), _row("a", 1, site="b")]

    snips = build_snips(rows, [_record("a")])

    assert [snip.site_id for snip in snips] == ["b", "z"]
    assert [_span(snip) for snip in snips] == [_expected(PAIRS, 1), _expected(PAIRS, 5)]


def test_two_rows_with_one_id_are_refused():
    with pytest.raises(ValueError, match="id"):
        build_snips([_row("a", 1, site="x"), _row("a", 2, site="x")], [_record("a")])


# --- the file --------------------------------------------------------------------------------


def _lines(path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_the_file_is_one_sorted_row_per_id_that_reads_back(tmp_path):
    snips = [Snip("b", _sha("a"), 100, 900, ALIGNMENT), Snip("a", _sha("b"), None, None, WHOLE_CLIP)]
    path = tmp_path / "snips.jsonl"

    write_snips(snips, path)

    assert _lines(path) == [
        {"id": "a", "audio_sha256": _sha("b"), "start_sample": None, "end_sample": None,
         "snip_source": "whole_clip"},
        {"id": "b", "audio_sha256": _sha("a"), "start_sample": 100, "end_sample": 900,
         "snip_source": "alignment"},
    ]
    assert read_snips(path) == {snip.site_id: snip for snip in snips}


@pytest.mark.parametrize("row", [
    {"id": "a", "audio_sha256": "x", "start_sample": 5, "end_sample": 5, "snip_source": "alignment"},
    {"id": "a", "audio_sha256": "x", "start_sample": 9, "end_sample": 5, "snip_source": "alignment"},
    {"id": "a", "audio_sha256": "x", "start_sample": None, "end_sample": None,
     "snip_source": "alignment"},
    {"id": "a", "audio_sha256": "x", "start_sample": 0, "end_sample": 5, "snip_source": "whole_clip"},
    {"id": "a", "audio_sha256": "x", "start_sample": None, "end_sample": None,
     "snip_source": "model"},
    {"id": "a", "audio_sha256": "x", "start_sample": 0, "snip_source": "alignment"},
])
def test_a_malformed_snip_row_is_refused(tmp_path, row):
    path = tmp_path / "snips.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_snips(path)


def test_two_snips_for_one_id_are_refused(tmp_path):
    row = {"id": "a", "audio_sha256": "x", "start_sample": None, "end_sample": None,
           "snip_source": "whole_clip"}
    path = tmp_path / "snips.jsonl"
    path.write_text(json.dumps(row) + "\n" + json.dumps(row) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="a"):
        read_snips(path)


# --- the CLI ---------------------------------------------------------------------------------


def _write_gz(path, records) -> None:
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        handle.writelines(json.dumps(record) + "\n" for record in records)


def _setup(tmp_path):
    """Six worklist rows: three that place (one in the second word-times file), and one each of
    not aligned, below the gate and no record at all."""
    rows = [_row("a", 1), _row("a", 5), _row("b", 3), _row("c", 3), _row("d", 3), _row("e", 3)]
    first = tmp_path / "confirm.1.jsonl.gz"
    second = tmp_path / "mined.confirm.2.jsonl.gz"
    _write_gz(first, [_record("a"), _record("c", aligned=False),
                      _record("d", ll=MIN_LL_PER_FRAME - 1.0)])
    _write_gz(second, [_record("b")])
    worklist = tmp_path / "worklist.jsonl"
    worklist.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    return rows, worklist, [first, second]


def _build(tmp_path, worklist, word_times, *extra, out="snips.jsonl"):
    argv = ["build", "--worklist", str(worklist), "--word-times", *map(str, word_times),
            "--out", str(tmp_path / out), *extra]
    main(argv)
    return tmp_path / out


def test_the_cli_builds_the_sidecar_from_gzipped_word_times_across_files(tmp_path):
    rows, worklist, word_times = _setup(tmp_path)

    snips = read_snips(_build(tmp_path, worklist, word_times))

    assert set(snips) == {row["id"] for row in rows}
    assert _span(snips["a-1"]) == _expected(PAIRS, 1)
    assert _span(snips["a-5"]) == _expected(PAIRS, 5)
    assert _span(snips["b-3"]) == _expected(PAIRS, 3)
    assert {snips[i].snip_source for i in ("c-3", "d-3", "e-3")} == {WHOLE_CLIP}


def test_the_cli_reports_the_fallback_rate_and_its_causes(tmp_path, capsys):
    _, worklist, word_times = _setup(tmp_path)
    report = tmp_path / "report.json"

    _build(tmp_path, worklist, word_times, "--report", str(report))

    data = json.loads(report.read_text(encoding="utf-8"))
    assert (data["rows"], data["alignment"], data["whole_clip"]) == (6, 3, 3)
    assert data["fallback_rate"] == pytest.approx(0.5)
    assert sum(data["reasons"].values()) == 3
    assert "fallback" in capsys.readouterr().out.lower()


def test_the_cli_takes_the_gate_as_a_flag(tmp_path):
    _, worklist, word_times = _setup(tmp_path)

    snips = read_snips(_build(tmp_path, worklist, word_times,
                              "--min-ll-per-frame", str(-0.1)))

    # The planted record's ll_per_frame is -0.2, below -0.1: every placed word now plays whole.
    assert {snip.snip_source for snip in snips.values()} == {WHOLE_CLIP}


def test_the_file_does_not_depend_on_the_worklist_files_order(tmp_path):
    rows, worklist, word_times = _setup(tmp_path)
    first = _build(tmp_path, worklist, word_times).read_bytes()
    worklist.write_text("".join(json.dumps(row) + "\n" for row in reversed(rows)),
                        encoding="utf-8")

    assert _build(tmp_path, worklist, word_times, out="again.jsonl").read_bytes() == first


def test_a_worklist_row_without_its_clip_checksum_is_refused(tmp_path):
    rows, worklist, word_times = _setup(tmp_path)
    broken = {name: value for name, value in rows[0].items() if name != "audio_sha256"}
    worklist.write_text(json.dumps(broken) + "\n", encoding="utf-8")

    with pytest.raises((ValueError, KeyError)):
        _build(tmp_path, worklist, word_times)

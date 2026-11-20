"""``label_of``: which label a tlog item is read against (#300, brief changelog (a)).

Round 2 grades the filename ayah (``training.tlog_eval``). The mined rows are read against their
``assigned_label`` (#263), so the item builders gain one optional keyword::

    gating_items(rows, lines, *, reference_of, canonical_words=None, label_of=<filename_label>)
    reported_item_sets(rows, lines, *, reference_of, canonical_words=None, label_of=<filename_label>)

``label_of(row) -> str | None`` names the ayah (``"surah:ayah"``) a manifest row is graded on;
``None`` leaves the row out as unsupported, like a row naming no ayah. The default is today's
``row["filename_label"]``: ``test_consonant_readout_tlog_items.py`` stays the pin for it and these
tests only add the new keyword.
"""

from __future__ import annotations

from training import tlog_eval
from training.nonpro_decode import item_key

AYAH_WORDS = {"2:3": 7, "2:5": 4}


def _canonical(label: str) -> list[str]:
    return [f"w{i}" for i in range(AYAH_WORDS[label])]


def _reference_of(words):
    return " ".join(words), [4 * i for i in range(len(words))]


KW = {"canonical_words": _canonical, "reference_of": _reference_of}


def _row(n: int, *, filename: str, assigned: str | None, stratum: str) -> dict:
    name = f"{filename.replace(':', '_')}_{n:09d}.flac"
    return {"split": "unclean", "shard_file": f"unclean-{n:05d}-of-00332", "row": n,
            "audio_path": name, "audio_sha256": f"{n:064x}", "group_key": name,
            "filename_label": filename, "assigned_label": assigned,
            "label_source": "identified", "straddles": False, "decode_mode": "whole_clip",
            "id_floor_ok": False, "stratum": stratum}


def _lines(rows) -> dict:
    return {item_key(r["audio_path"]): {
        "key": item_key(r["audio_path"]), "audio_filename": r["audio_path"],
        "audio_sha256": r["audio_sha256"], "surah_ayah": None, "word_from": None,
        "word_to": None, "start_sample": 0, "end_sample": 32000, "split": r["split"],
        "shard_file": r["shard_file"], "row": r["row"], "stratum": r["stratum"],
        "assigned_label": r["assigned_label"], "checks": []} for r in rows}


def _assigned(row) -> str | None:
    return row["assigned_label"]


def test_gating_items_read_the_label_label_of_names():
    row = _row(1, filename="2:3", assigned="2:5", stratum="gating")
    (item,), unsupported = tlog_eval.gating_items([row], _lines([row]), label_of=_assigned, **KW)
    assert (item.surah, item.ayah) == (2, 5)
    assert item.reference.words == tuple(_canonical("2:5"))
    assert unsupported == []


def test_reported_item_sets_read_the_label_label_of_names():
    rows = [_row(1, filename="2:3", assigned="2:5", stratum="reported:identified"),
            _row(2, filename="2:3", assigned="2:3", stratum="reported:below_bar")]
    sets = tlog_eval.reported_item_sets(rows, _lines(rows), label_of=_assigned, **KW)
    (identified,) = sets["reported:identified"]
    (below_bar,) = sets["reported:below_bar"]
    assert (identified.surah, identified.ayah) == (2, 5)
    assert identified.reference.word_indices == tuple(range(4))
    assert (below_bar.surah, below_bar.ayah) == (2, 3)


def test_a_row_whose_label_is_none_is_unsupported_not_graded():
    row = _row(1, filename="2:3", assigned=None, stratum="reported:identified")
    sets = tlog_eval.reported_item_sets([row], _lines([row]), label_of=_assigned, **KW)
    assert sets["reported:identified"] == []
    items, unsupported = tlog_eval.gating_items(
        [{**row, "stratum": "gating"}], _lines([{**row, "stratum": "gating"}]),
        label_of=_assigned, **KW)
    assert items == [] and unsupported == [item_key(row["audio_path"])]


def test_without_label_of_the_filename_ayah_is_still_what_is_graded():
    row = _row(1, filename="2:3", assigned="2:5", stratum="gating")
    (item,), _ = tlog_eval.gating_items([row], _lines([row]), **KW)
    assert (item.surah, item.ayah) == (2, 3)

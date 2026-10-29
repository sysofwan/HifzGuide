"""tlog items for the round-2 readout (#227): gating only, the filename ayah in full, surah clusters.

The interface these tests pin, in ``training.tlog_eval`` (new)::

    gating_items(rows, lines, *, canonical_words, reference_of) -> (items, unsupported)
    span_trimmed_items(rows, lines, *, canonical_words, reference_of) -> (items, unsupported)
    reported_item_sets(rows, lines, *, canonical_words, reference_of) -> {name: items}
    confirm_views({"confirm.1": items, "confirm.2": items}) -> {"tlog.confirm": pooled, ...}

``rows`` are a tranche manifest's rows (``tadabur.tlog_tranches.MANIFEST_FIELDS``), ``lines`` the
committed decode records by key, ``canonical_words(label)`` the ayah's canonical words (their
count is N, from the reference database), ``reference_of(words)`` the phonetizer. Items are
:class:`training.nonpro_eval.ClipItem` s.
"""

from __future__ import annotations

import pytest

from tadabur.tlog_index import read_jsonl
from tadabur.tlog_tranches import TRANCHES_DIR
from training.acceptance_stats import Resample
from training.nonpro_decode import item_key

AYAH_WORDS = {"2:3": 7, "2:5": 4, "101:4": 5, "103:3": 9, "1:2": 3}


def _canonical(label: str) -> list[str]:
    return [f"w{i}" for i in range(AYAH_WORDS[label])]


def _reference_of(words):
    return " ".join(words), [4 * i for i in range(len(words))]


KW = {"canonical_words": _canonical, "reference_of": _reference_of}


def _row(n: int, label: str = "2:3", stratum: str = "gating", *, assigned: str | None = None,
         id_floor_ok: bool = True) -> dict:
    name = f"{label.replace(':', '_')}_{n:09d}.flac"
    return {"split": "clean", "shard_file": f"clean-{n:05d}-of-00411", "row": n,
            "audio_path": name, "audio_sha256": f"{n:064x}", "group_key": name,
            "filename_label": label, "assigned_label": assigned or label,
            "label_source": "both", "straddles": False, "decode_mode": "whole_clip",
            "id_floor_ok": id_floor_ok, "stratum": stratum}


def _line(row: dict, span: tuple[int, int] | None = None) -> dict:
    """A committed decode line: #212's merge wrote S6's identified span as word_from/word_to."""
    words = AYAH_WORDS[row["filename_label"]]
    first, last = span or (1, words)
    return {"key": item_key(row["audio_path"]), "audio_filename": row["audio_path"],
            "audio_sha256": row["audio_sha256"], "surah_ayah": row["filename_label"],
            "word_from": first, "word_to": last, "start_sample": 0, "end_sample": 32000,
            "split": row["split"], "shard_file": row["shard_file"], "row": row["row"],
            "stratum": row["stratum"], "assigned_label": row["assigned_label"], "checks": []}


def _lines(rows, spans=None) -> dict:
    spans = spans or {}
    return {line["key"]: line for line in (_line(r, spans.get(r["row"])) for r in rows)}


# --- gating only, the whole filename ayah ----------------------------------------------------------


def test_only_gating_rows_become_the_gating_items():
    rows = [_row(1), _row(2), _row(3, stratum="reported:below_bar"),
            _row(4, stratum="reported:unclean"), _row(5, stratum="reported:undecided")]
    items, unsupported = _gating(rows)
    assert [i.audio_filename for i in items] == [r["audio_path"] for r in rows[:2]]
    assert unsupported == []


def _gating(rows, spans=None, **kw):
    from training import tlog_eval

    return tlog_eval.gating_items(rows, _lines(rows, spans), **{**KW, **kw})


def test_an_item_carries_its_clip_its_decode_span_and_the_decode_s_key():
    row = _row(7, "2:5")
    (item,), _ = _gating([row])
    assert item.key == item_key(row["audio_path"])
    assert (item.audio_filename, item.audio_sha256) == (row["audio_path"], row["audio_sha256"])
    assert (item.surah, item.ayah) == (2, 5)
    assert (item.start_sample, item.end_sample) == (0, 32000)


def test_the_reference_is_the_filename_ayah_in_full_despite_a_planted_span():
    plain, planted = _row(1, "103:3"), _row(2, "103:3")
    items, _ = _gating([plain, planted], spans={2: (2, 5)})
    for item in items:
        assert item.reference.words == tuple(_canonical("103:3"))
        assert item.reference.word_indices == tuple(range(9))
    assert items[0].reference == items[1].reference


def test_the_filename_label_decides_the_ayah_not_the_assigned_label():
    row = _row(1, "2:3", assigned="2:5")
    (item,), _ = _gating([row])
    assert (item.surah, item.ayah) == (2, 3)
    assert len(item.reference.words) == AYAH_WORDS["2:3"]


def test_the_word_count_comes_from_the_canonical_words_not_from_the_decode_span():
    row = _row(1, "2:3")
    lines = _lines([row], spans={1: (1, 2)})
    lines[item_key(row["audio_path"])]["word_to"] = 99
    from training import tlog_eval

    (item,), _ = tlog_eval.gating_items([row], lines, **KW)
    assert len(item.reference.words) == AYAH_WORDS["2:3"]


def test_a_gating_row_with_no_decode_is_refused_rather_than_dropped():
    from training import tlog_eval

    rows = [_row(1), _row(2)]
    lines = _lines(rows[:1])
    with pytest.raises(ValueError, match="decode"):
        tlog_eval.gating_items(rows, lines, **KW)


def test_a_clip_the_phonetizer_cannot_handle_is_listed_not_graded():
    def reference_of(words):
        if len(words) == AYAH_WORDS["2:5"]:
            raise KeyError("no phonemes")
        return _reference_of(words)

    rows = [_row(1, "2:3"), _row(2, "2:5")]
    items, unsupported = _gating(rows, reference_of=reference_of)
    assert [i.key for i in items] == [item_key(rows[0]["audio_path"])]
    assert unsupported == [item_key(rows[1]["audio_path"])]


def test_the_items_come_back_in_key_order_whatever_the_manifest_order():
    rows = [_row(3, "2:5"), _row(1, "2:3"), _row(2, "101:4")]
    items, _ = _gating(rows)
    assert [i.key for i in items] == sorted(i.key for i in items)


# --- clusters --------------------------------------------------------------------------------------


def test_the_bootstrap_cluster_is_the_surah():
    rows = [_row(1, "2:3"), _row(2, "2:5"), _row(3, "101:4"), _row(4, "103:3")]
    items, _ = _gating(rows)
    assert {i.audio_filename: i.cluster for i in items} == {
        rows[0]["audio_path"]: 2, rows[1]["audio_path"]: 2, rows[2]["audio_path"]: 101,
        rows[3]["audio_path"]: 103}
    assert Resample.by_cluster([i.cluster for i in items]).clusters == (2, 101, 103)


# --- the span-trimmed sensitivity, and the reported strata ----------------------------------------


def test_the_span_trimmed_variant_reads_the_identified_span_and_only_as_a_sensitivity():
    from training import tlog_eval

    rows = [_row(1, "103:3"), _row(2, "103:3")]
    lines = _lines(rows, spans={2: (2, 5)})
    trimmed, _ = tlog_eval.span_trimmed_items(rows, lines, **KW)
    gating, _ = tlog_eval.gating_items(rows, lines, **KW)
    assert [i.key for i in trimmed] == [i.key for i in gating]
    assert len(trimmed[0].reference.words) == 9  # a clip with no partial span is unchanged
    assert trimmed[1].reference.words == tuple(_canonical("103:3")[1:5])
    assert trimmed[1].reference.word_indices == (1, 2, 3, 4)
    assert len(gating[1].reference.words) == 9


def test_reported_strata_are_their_own_sets_and_never_enter_gating():
    from training import tlog_eval

    rows = [_row(1), _row(2, id_floor_ok=False), _row(3, stratum="reported:below_bar"),
            _row(4, stratum="reported:below_bar"), _row(5, stratum="reported:unclean")]
    lines = _lines(rows)
    sets = tlog_eval.reported_item_sets(rows, lines, **KW)
    assert set(sets) == {"gating.id_floor_ok", "reported:below_bar", "reported:unclean"}
    key = lambda row: item_key(row["audio_path"])  # noqa: E731
    assert [i.key for i in sets["gating.id_floor_ok"]] == [key(rows[0])]
    assert [i.key for i in sets["reported:below_bar"]] == sorted([key(rows[2]), key(rows[3])])
    gating, _ = tlog_eval.gating_items(rows, lines, **KW)
    assert {i.key for i in gating} == {key(rows[0]), key(rows[1])}
    assert not {i.key for i in gating} & {i.key for name, items in sets.items()
                                          if name.startswith("reported:") for i in items}


# --- confirm.1 and confirm.2 pooled, each tranche still read alone ---------------------------------


def _tranche(start: int, labels: list[str]):
    from training import tlog_eval

    rows = [_row(start + n, label) for n, label in enumerate(labels)]
    return tlog_eval.gating_items(rows, _lines(rows), **KW)[0]


def test_confirm_one_and_two_are_pooled_into_one_set_with_a_view_of_each():
    from training import tlog_eval

    one = _tranche(100, ["2:3", "2:5", "101:4"])
    two = _tranche(200, ["2:3", "103:3"])
    views = tlog_eval.confirm_views({"confirm.1": one, "confirm.2": two})
    assert set(views) == {"tlog.confirm", "confirm.1", "confirm.2"}
    assert views["confirm.1"] == one and views["confirm.2"] == two
    pooled = views["tlog.confirm"]
    assert [i.key for i in pooled] == sorted(i.key for i in [*one, *two])
    assert len(pooled) == 5


def test_a_surah_in_both_tranches_is_one_cluster_of_the_pooled_resample():
    from training import tlog_eval

    one = _tranche(100, ["2:3", "101:4"])
    two = _tranche(200, ["2:5", "103:3"])
    views = tlog_eval.confirm_views({"confirm.1": one, "confirm.2": two})
    pooled = Resample.by_cluster([i.cluster for i in views["tlog.confirm"]]).clusters
    alone = [Resample.by_cluster([i.cluster for i in views[name]]).clusters
             for name in ("confirm.1", "confirm.2")]
    assert pooled == (2, 101, 103)
    assert sum(len(c) for c in alone) == 4  # resampled apart, surah 2 would be two clusters


def test_only_confirm_one_and_two_can_be_pooled_and_a_row_cannot_be_in_both():
    from training import tlog_eval

    one = _tranche(100, ["2:3"])
    for tranches in ({"confirm.3": one}, {"confirm.1": one, "confirm.4": one},
                     {"confirm.2": one}, {"look": one}):
        with pytest.raises(ValueError):
            tlog_eval.confirm_views(tranches)
    with pytest.raises(ValueError, match="twice|both"):
        tlog_eval.confirm_views({"confirm.1": one, "confirm.2": one})


# --- the real look manifest ------------------------------------------------------------------------


def test_real_look_clips_with_a_partial_span_are_graded_on_their_whole_ayah():
    """#212's committed decodes carry S6's span as word_from/word_to; the item ignores it."""
    import gzip
    import json

    from tadabur.waqf_segments import _uthmani_words, hafs_segment_reference
    from training import tlog_eval

    decodes = TLOG_LOOK_DECODES / "tlog.look.c95.jsonl.gz"
    with gzip.open(decodes, "rt", encoding="utf-8") as stream:
        spanned = [line for line in map(json.loads, stream) if line["word_from"] != 1][:4]
    assert len(spanned) == 4
    manifest = {(r["split"], r["shard_file"], r["row"]): r
                for r in read_jsonl(TRANCHES_DIR / "look.jsonl")}
    rows = [manifest[(line["split"], line["shard_file"], line["row"])] for line in spanned]
    items, unsupported = tlog_eval.gating_items(
        rows, {line["key"]: line for line in spanned}, canonical_words=_uthmani_words,
        reference_of=hafs_segment_reference())
    assert len(items) + len(unsupported) == 4 and items
    for item in items:
        words = _uthmani_words(item.surah_ayah)
        assert item.reference.words == tuple(words)
        assert item.reference.word_indices == tuple(range(len(words)))
        assert item.cluster == item.surah


TLOG_LOOK_DECODES = TRANCHES_DIR.parent.parent / "consonant2_readout" / "decodes"


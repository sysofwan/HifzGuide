"""The S6 audit draw's item shape (``tadabur.tlog_ident_run.audit_draw``).

``stage`` copies a fixed key set off every drawn item to build the worklist the page and the
report read, so a key present on only one stratum crashes the staging run after it has already
downloaded shards. Every drawn item carries the whole set, ``blocking_reason`` included.
"""

from __future__ import annotations

from . import tlog_ident_run as run

#: The keys ``stage`` reads off a drawn item.
WORKLIST_KEYS = {"population", "split", "shard_file", "row", "role", "audio_path",
                 "audio_sha256", "duration_s", "filename_label", "assigned_label",
                 "class_label", "label_source", "id_floor_ok", "overlaps_filename",
                 "blocking_reason", "stratum"}


def _row(split: str, row: int, **fields) -> dict:
    return {"population": "look", "role": "look", "split": split,
            "shard_file": f"{split}-00000-of-00332", "row": row,
            "audio_path": f"audio/{split}-00000-of-00332.parquet", "audio_sha256": "0" * 64,
            "duration_s": 12.5, "filename_label": "2:1", "assigned_label": "2:1",
            "class_label": "2:1", "label_source": "filename", "id_floor_ok": True,
            "overlaps_filename": False, "decode_mode": "whole_clip", **fields}


def _records() -> list[dict]:
    return [
        _row("clean", 1, label_source="both"),
        _row("clean", 2, label_source="identified"),
        _row("unclean", 3, label_source="identified"),
        _row("clean", 4, id_floor_ok=False),
        _row("clean", 5, class_label="2:2"),
        _row("clean", 6, exclusions=("silence",), label_source="none", id_floor_ok=None,
             class_label=None, assigned_label=None),
    ]


def test_every_drawn_stratum_carries_the_whole_worklist_key_set():
    draw = run.audit_draw(_records(), sizes=dict.fromkeys(run.AUDIT_SIZES, 1))
    strata = {item["stratum"] for item in draw.items}
    assert strata == {"both", "relabelled", "unclean identified", "non_recitation",
                      "below-bar disagreement", "silence"}
    for item in draw.items:
        assert WORKLIST_KEYS <= set(item), item["stratum"]
    reasons = {item["stratum"]: item["blocking_reason"] for item in draw.items}
    assert reasons["below-bar disagreement"] in ("overlap", "delta")
    assert all(reason is None for stratum, reason in reasons.items()
               if stratum != "below-bar disagreement")

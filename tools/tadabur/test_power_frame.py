"""Ship A's decode-level frame (#126): strata, fields, and the committed frame. Torch-free."""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

from tadabur.mining_pool import PoolClip, PoolSegment
from tadabur.power_frame import (
    FIELDS,
    FRAME_PATH,
    INPUTS,
    build_frame,
    input_checksums,
    score_pool,
    site_fields,
)
from tadabur.staged_audio import StagedClip

FATHA, KASRA = "َ", "ِ"
#: كَذَببِتَم: a ذ carrier (in ذ↔ز and ذ↔ظ), an in-word geminate بب, a ت carrier.
REFERENCE = f"ك{FATHA}ذ{FATHA}بب{KASRA}ت{FATHA}م"
#: h448 commits the partner ز at ذ and holds the geminate as one ب; the base gets both right.
H448 = f"ك{FATHA}ز{FATHA}ب{KASRA}ت{FATHA}م"
FIELD = {name: i for i, name in enumerate(FIELDS)}


def scored_pool(h448: str = H448, base: str = REFERENCE, reference: str = REFERENCE):
    segment = PoolSegment(0, 0, 1, 16000, 64000, reference, (0, len(reference)), True)
    clip = PoolClip("c.wav", ("uniform",), 0.5, "2:2", 3, 1, None, 0, 1, 0.0, 4.0,
                    (1.0, 3.0), (segment,))
    registry = {"c.wav": StagedClip("c.wav", 77, 5, 3, "2:2", 80000, "b" * 64,
                                    ("mining_pool",))}
    return score_pool([clip], {"c.wav#0": base}, {"c.wav#0": h448}, registry,
                      lambda _: [False])


def test_strata_follow_the_reference_alone():
    strata = {(one.stratum, one.site.reference_index) for one in scored_pool()}
    assert {("pair:ذ→ز", 2), ("pair:ذ→ظ", 2), ("pair:ت→ط", 7), ("pair:ك→ق", 0),
            ("shaddah:held:in_word", 4)} <= strata
    assert strata == {(one.stratum, one.site.reference_index)
                      for one in scored_pool(h448=REFERENCE)}


def test_the_fields_read_each_decode_against_the_mushaf():
    by = {one.stratum: one for one in scored_pool() if one.site.reference_index in (2, 4)}
    dhal_zay, dhal_zah, geminate = (site_fields(by[s]) for s in (
        "pair:ذ→ز", "pair:ذ→ظ", "shaddah:held:in_word"))
    for fields in (dhal_zay, dhal_zah):
        assert fields[FIELD["h448_commit"]] and not fields[FIELD["h448_mushaf"]]
        assert fields[FIELD["base_mushaf_where_h448_flagged"]]
        assert not fields[FIELD["base_agrees_with_h448"]]
    assert dhal_zay[FIELD["h448_partner"]] and not dhal_zah[FIELD["h448_partner"]]
    assert geminate[FIELD["h448_single_on_held"]]
    assert geminate[FIELD["base_mushaf_where_h448_flagged"]]


def test_the_frame_sums_every_field_per_stratum_and_reciter():
    frame = build_frame(scored_pool())
    entry = frame["pair:ذ→ز"]
    assert (entry["family"], entry["mode"], entry["population"]) == ("ذ↔ز", "consonant", 1)
    sums = entry["per_reciter"]["3"]
    assert sums[FIELD["sites"]] == 1 and sums[FIELD["h448_partner"]] == 1
    assert frame["shaddah:held:in_word"]["family"] == "shaddah"


def test_no_field_comes_from_an_app_model():
    """The frame is decode level: the Muraja policy module is never imported."""
    for module in ("power_frame.py", "ship_a.py", "word_sample.py", "ship_a_worklist.py"):
        tree = ast.parse((Path(__file__).parent / module).read_text(encoding="utf-8"))
        imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        assert not any("muraja_policy" in (name or "") for name in imported), module


def test_the_committed_frame_was_built_from_the_committed_inputs():
    frame = json.loads(FRAME_PATH.read_text(encoding="utf-8"))
    assert frame["inputs"] == input_checksums(INPUTS)
    assert tuple(frame["fields"]) == FIELDS
    geminates = frame["strata"]["shaddah:held:in_word"]
    totals = [sum(v[i] for v in geminates["per_reciter"].values()) for i in range(len(FIELDS))]
    assert geminates["population"] == totals[FIELD["sites"]] == 8480


def test_the_frame_needs_no_torch():
    code = ("import sys; import tadabur.power_frame, tadabur.ship_a, tadabur.word_sample; "
            "assert 'torch' not in sys.modules")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=Path(__file__).parent.parent)

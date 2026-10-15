"""Tests for training.external_corpus beyond what QUD's pilot tests: every consonant (#156).

Torch-free and model-free: the items and decodes are synthetic. #144's rule, the sites, the
worklist, the estimate and the app outcome are tested through QUD in ``test_qud_pilot.py``.
"""

from __future__ import annotations

from tadabur.qud_text import ItemReference
from training import external_corpus as ec
from training.nonpro_eval import ClipItem

# ``قَالَ سَططَ``: carriers ق, ل, س and the geminate طط (one carrier, on its first half).
REFERENCE = "قَالَ سَططَ"
OFFSETS = (0, 6, 11)


def _item(key: str = "c#0") -> ClipItem:
    return ClipItem(key, "qdat", "c", 1, 1, 0, 32000, "e" * 64,
                    ItemReference(("a", "b"), (0, 1), REFERENCE, OFFSETS), "s1", 0)


def _carriers(base: str, h448: str):
    item = _item()
    decodes = {"base": {item.key: {"spans": base}}, "h448": {item.key: {"stream_b0": h448}}}
    return ec.consonant_carriers([item], decodes, workers=1)


def test_every_consonant_run_is_one_carrier_read_by_the_two_rule_arms():
    carriers = _carriers(REFERENCE, "قَالَ شَططَ")  # h448 hears the س as ش
    assert [(c.reference_index, c.prescribed) for c in carriers] == \
        [(0, "ق"), (3, "ل"), (6, "س"), (8, "ط")]
    assert [c.cls for c in carriers] == [ec.AGREED, ec.AGREED, ec.DISAGREED, ec.AGREED]


def test_consonant_counts_tally_classes_and_what_each_arm_committed_instead():
    counts = ec.consonant_counts(_carriers("قَالَ صَططَ", "قَالَ شَططَ"))
    assert counts["letters"]["س"] == {"carriers": 1, ec.DISAGREED: 1}
    assert counts["total"] == {"carriers": 4, ec.AGREED: 3, ec.DISAGREED: 1}
    assert counts["committed_instead"] == {
        "base/spans": [{"substitution": "س→ص", "carriers": 1}],
        "h448/stream_b0": [{"substitution": "س→ش", "carriers": 1}]}


def test_nothing_committed_is_abstained_and_never_a_worklist_row():
    carriers = _carriers("قَالَ", "قَالَ شَططَ")  # base never reaches the second word
    item = _item()
    assert {c.prescribed: c.cls for c in carriers}["س"] == ec.DISAGREED
    assert {c.prescribed: c.cls for c in _carriers("قَالَ", REFERENCE)}["ط"] == ec.ABSTAINED
    work = ec.consonant_worklist(carriers, {item.key: item})
    assert [(r["carrier"], r["stratum"], r["committed"]) for r in work] == [
        ("qdat:c#0:6", "consonant:س", {"base/spans": None, "h448/stream_b0": "ش"})]
    assert (work[0]["excerpt_start_sample"], work[0]["excerpt_end_sample"]) == (0, 32000)
    assert work[0]["audio_sha256"] == "e" * 64  # a clip names its audio's checksum


def test_a_clip_item_is_a_corpus_item_with_its_dataset_as_source():
    item = _item()
    assert (item.source, item.surah_ayah, item.cluster) == ("qdat", "1:1", 0)
    assert item.provenance == {"audio_filename": "c", "audio_sha256": "e" * 64}
    clip = ec.clip_of(item, {m: {item.key: {"spans": REFERENCE, "stream_b0": REFERENCE,
                                            "checks": [{"hop": REFERENCE}]}}
                             for m in ("base", "h448")})
    # The pair carriers ق, س and ط, and the geminate طط: ل is in no target pair.
    assert clip.reciter_id == 0 and len(clip.positions) == 4

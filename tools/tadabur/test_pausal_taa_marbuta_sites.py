"""Ship A's site derivation and the guard apply the frozen pause corrections (#130).

A synthetic 2:254 segment: the reference has شَفَاعَةٌ in wasl ت, and the decode says the
pausal ه. With the correction cut after the word, the reference carries ه, so the carrier
is no longer the prescribed ت and the taa site is not mined.
"""

from __future__ import annotations

from tadabur import waqf_guard as g
from tadabur.listening_session import pool_candidates
from tadabur.mining_pool import PoolClip, PoolSegment
from tadabur.power_frame import score_pool
from tadabur.ship_a import mine_all
from tadabur.staged_audio import StagedClip
from tadabur.waqf_segments import _uthmani_words, hafs_segment_reference

NAME = "2:254.wav"
CUT = 18  # after شَفَاعَةٌ in 2:254: the pause the guard heard
WORDS = 21
CLIP_SAMPLES = 25 * 16000


def _fixture():
    words = _uthmani_words("2:254")
    wasl, offsets = hafs_segment_reference()(words)
    assert len(words) == WORDS
    base = wasl.replace(" ", "")
    h448 = base.replace("شَفَااعَتُوو", "شَفَااعَه")  # the reciter paused: ه
    segment = PoolSegment(0, 0, len(words), 0, CLIP_SAMPLES, wasl, tuple(offsets), True)
    clip = PoolClip(NAME, ("uniform",), 1.0, "2:254", 0, len(words), None, 0, len(words),
                    0.0, 25.0, (), (segment,))
    registry = {NAME: StagedClip(NAME, 0, 0, 0, "2:254", CLIP_SAMPLES, "a" * 64,
                                 ("mining_pool",))}
    return base, h448, clip, registry


def _tanween(_surah_ayah: str) -> list[bool]:
    return [False] * WORDS


def _local_word(offsets, index: int) -> int:
    return max(word for word, offset in enumerate(offsets) if offset <= index)


def test_classify_realizes_an_item_with_its_cuts():
    from tadabur.waqf_segments import hafs_paused_reference

    words = _uthmani_words("2:254")
    reference, offsets = hafs_paused_reference(words, (CUT,))
    carrier = reference.index("ه", offsets[17], offsets[18])
    item = g.Item("2:254@18", ("test",), "new_audit", "test", "ت↔ط", "ت", NAME, "2:254",
                  0, tuple(offsets), reference, carrier, 0, CLIP_SAMPLES, 0, cuts=(CUT,))
    found = g.classify(item)
    assert found.dependence == g.NOT_DEPENDENT
    assert not any(dependence.word == CUT for dependence in found.dependences)


def test_a_paused_taa_marbuta_site_is_not_mined():
    base, h448, clip, registry = _fixture()
    decodes = {f"{NAME}#0": base}
    cuts = {(NAME, 0): (CUT,)}
    wasl = pool_candidates([clip], decodes, {f"{NAME}#0": h448}, registry, _tanween, cuts={})
    assert any(
        candidate.site.mark == "ت↔ط" and _local_word(candidate.word_offsets,
                                                     candidate.site.reference_index) == 17
        for candidate in wasl
    )
    corrected = pool_candidates([clip], decodes, {f"{NAME}#0": h448}, registry, _tanween,
                                cuts=cuts)
    assert not any(
        candidate.site.mark == "ت↔ط" and _local_word(candidate.word_offsets,
                                                     candidate.site.reference_index) == 17
        for candidate in corrected
    )
    scored = score_pool([clip], decodes, {f"{NAME}#0": h448}, registry, _tanween, cuts=cuts)
    mined = mine_all(scored)
    assert not any(
        site.site.mark == "ت↔ط"
        and _local_word(site.row.word_offsets, site.site.reference_index) == 17
        for site in mined
    )

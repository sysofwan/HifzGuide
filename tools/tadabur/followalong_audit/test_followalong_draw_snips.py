"""The draw's snip file: #264's rule, with S5's edge words played whole (#311).

Committed before ``tadabur/followalong_audit/draw.py`` exists and **locked**: later turns may add
tests but never weaken or rewrite these. They pin the preregistration §3 ("an S5 leading or
trailing word is served as a whole clip") on top of :mod:`tadabur.word_snips`.

Contract this file pins (``tadabur.followalong_audit.draw``)
-----------------------------------------------------------
``snips_for_draw(worklist, key, records) -> list[Snip]``:

* ``worklist`` rows are ``tadabur.catch_loss_worklist.WORKLIST_FIELDS`` dicts (``end_sample`` is
  the decoded 16 kHz clip's length); ``key`` rows carry at least ``id`` and ``whole_played``;
  ``records`` are phase A word-time records.
* One :class:`tadabur.word_snips.Snip` per worklist row, sorted by id, built by
  :func:`tadabur.word_snips.build_snips` at its ``ll_per_frame`` gate (-3.5) — **except** that a row
  whose key says ``whole_played`` is a ``WHOLE_CLIP`` snip with no bounds, whatever its alignment.
* A repeat (a worklist row with its own id and its original's clip and word) gets the same bounds
  as the original. A worklist row with no key row is refused (``ValueError``): the blind file and
  the key must describe the same items.
"""

from __future__ import annotations

import pytest

from tadabur.followalong_audit.draw import snips_for_draw
from tadabur.word_snips import ALIGNMENT, WHOLE_CLIP, build_snips

CLIP_SAMPLES = 16_000 * 20


def record(digest: str, *, ayah="2:1", ll=-0.5, aligned=True, words=8) -> dict:
    return {"audio_sha256": digest, "surah_ayah": ayah, "aligned": aligned, "ll_per_frame": ll,
            "word_times": [[1.0 + n, 1.8 + n] for n in range(words)]}


def row(site_id: str, digest: str, word_index=3, ayah="2:1") -> dict:
    return {"id": site_id, "split": "clean", "shard_file": "clean-00000-of-00411", "row": 1,
            "audio_filename": f"{digest[:4]}.flac", "audio_sha256": digest, "start_sample": 0,
            "end_sample": CLIP_SAMPLES, "surah_ayah": ayah, "word_index": word_index,
            "word": "كلمة"}


def key_row(site_id: str, *, whole=False) -> dict:
    return {"id": site_id, "whole_played": whole}


@pytest.fixture
def draw():
    worklist = [row("0000000000000001", "a" * 64), row("0000000000000002", "b" * 64),
                row("0000000000000003", "c" * 64), row("0000000000000004", "d" * 64),
                row("0000000000000005", "e" * 64)]
    key = [key_row("0000000000000001"), key_row("0000000000000002", whole=True),
           key_row("0000000000000003", whole=True), key_row("0000000000000004"),
           key_row("0000000000000005")]
    records = [record("a" * 64), record("b" * 64), record("c" * 64),
               record("d" * 64, ll=-4.0), record("e" * 64)]
    return worklist, key, records


def by_id(snips):
    return {snip.site_id: snip for snip in snips}


def test_a_word_the_alignment_places_plays_its_excerpt(draw):
    worklist, key, records = draw
    snips = by_id(snips_for_draw(worklist, key, records))
    expected = by_id(build_snips(worklist, records))["0000000000000001"]
    assert snips["0000000000000001"] == expected
    assert snips["0000000000000001"].snip_source == ALIGNMENT


def test_an_s5_edge_word_plays_whole_although_the_alignment_could_place_it(draw):
    worklist, key, records = draw
    assert all(snip.snip_source == ALIGNMENT
               for snip in build_snips(worklist, records)
               if snip.site_id in ("0000000000000002", "0000000000000003"))
    snips = by_id(snips_for_draw(worklist, key, records))
    for site_id in ("0000000000000002", "0000000000000003"):
        assert snips[site_id].snip_source == WHOLE_CLIP
        assert (snips[site_id].start_sample, snips[site_id].end_sample) == (None, None)


def test_a_word_below_the_likelihood_gate_still_plays_whole(draw):
    worklist, key, records = draw
    assert by_id(snips_for_draw(worklist, key, records))["0000000000000004"].snip_source == WHOLE_CLIP


def test_the_gate_is_the_owner_approved_minus_three_and_a_half(draw):
    worklist, key, records = draw
    records[4] = record("e" * 64, ll=-3.6)
    snips = by_id(snips_for_draw(worklist, key, records))
    assert snips["0000000000000005"].snip_source == WHOLE_CLIP
    records[4] = record("e" * 64, ll=-3.4)
    assert by_id(snips_for_draw(worklist, key, records))["0000000000000005"].snip_source == ALIGNMENT


def test_one_sorted_snip_per_worklist_row(draw):
    worklist, key, records = draw
    snips = snips_for_draw(list(reversed(worklist)), key, records)
    assert [snip.site_id for snip in snips] == sorted(row["id"] for row in worklist)


def test_a_repeat_gets_its_originals_excerpt_under_its_own_id(draw):
    worklist, key, records = draw
    worklist.append(row("00000000000000ff", "a" * 64))
    key.append(key_row("00000000000000ff"))
    snips = by_id(snips_for_draw(worklist, key, records))
    original, again = snips["0000000000000001"], snips["00000000000000ff"]
    assert again.snip_source == original.snip_source == ALIGNMENT
    assert (again.start_sample, again.end_sample) == (original.start_sample, original.end_sample)


def test_a_repeat_of_an_edge_word_also_plays_whole(draw):
    worklist, key, records = draw
    worklist.append(row("00000000000000fe", "b" * 64))
    key.append(key_row("00000000000000fe", whole=True))
    assert by_id(snips_for_draw(worklist, key, records))["00000000000000fe"].snip_source == WHOLE_CLIP


def test_a_worklist_row_with_no_key_row_is_refused(draw):
    worklist, key, records = draw
    with pytest.raises(ValueError):
        snips_for_draw(worklist, key[:-1], records)

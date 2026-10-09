"""Tests for mining the listening session and storing its verdicts (#87).

Everything runs on hand-built references, decodes and rows: no model, phonetizer or audio.
"""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from tadabur.listening_session import (
    DEFAULT_SIZES,
    STRATA,
    VERDICTS_PATH,
    WORKLIST_PATH,
    SUMMARY_PATH,
    Candidate,
    SessionSite,
    Verdict,
    adjudicated,
    carrier_marks,
    census,
    draw,
    excerpt_span,
    fetch_list,
    hearable,
    load_worklist,
    mode_of,
    p35_candidates,
    pool_candidates,
    read_verdicts,
    segment_sites,
    shuffled,
    site_id,
    summarize,
    tashkeel_stratum,
    word_limits,
    write_verdicts,
    write_worklist,
)
from tadabur.mining_pool import PoolClip, PoolSegment
from tadabur.staged_audio import StagedClip, load_staged_clips
from tadabur.truth_sites import PENDING, TruthSite

FATHA, DAMMA, KASRA = "َ", "ُ", "ِ"


def T(mark: str, base: str, h448: str | None = None) -> str:
    """A tashkeel stratum; the h448 outcome defaults to the base's."""
    return tashkeel_stratum(mark, base, h448 or base)


def _found(reference, decode, tanween=None, offsets=None, h448=None):
    """The (stratum, carrier) of every site; the h448 decode defaults to the base's."""
    offsets = offsets or [0, len(reference)]
    tanween = tanween or [False] * (len(offsets) - 1)
    found = segment_sites(reference, decode, decode if h448 is None else h448, offsets, tanween)
    return {(f.stratum, f.reference_index) for f in found}


def _tashkeel(found):
    return {f for f in found if f[0].split(":")[1] in ("fatha", "damma", "kasra", "sukun")}


# --- which sites are eligible, and which stratum the base decode puts them in -----------


def test_the_base_decode_says_what_it_emitted_after_each_matched_carrier():
    assert carrier_marks(f"ك{FATHA}تب", f"ك{FATHA}ت{FATHA}ب") == {0: FATHA, 2: None, 4: None}


def test_a_mid_word_haraka_is_stratified_by_what_each_decode_emitted_there():
    reference = f"ك{FATHA}ت{FATHA}ب"  # كَتَب: the fatha on ت is mid-word, ب is word-final
    assert _tashkeel(_found(reference, f"ك{FATHA}تب")) == {
        (T("fatha", "matched"), 0), (T("fatha", "empty"), 2)}
    assert (T("fatha", "other"), 2) in _found(reference, f"ك{FATHA}ت{KASRA}ب")


def test_the_h448_stream_is_a_second_axis_of_the_tashkeel_strata():
    # The base heard the fatha on ت; the shipped student left the slot empty.
    reference = f"ك{FATHA}ت{FATHA}ب"
    found = _found(reference, reference, h448=f"ك{FATHA}تب")
    assert (T("fatha", "matched", "empty"), 2) in found
    # Its text is cut from a whole-clip stream, so it may carry a neighbour's letters.
    padded = _found(reference, reference, h448=f"ملك{FATHA}تبلن")
    assert (T("fatha", "matched", "empty"), 2) in padded


def test_a_word_final_haraka_is_not_a_site():
    found = _tashkeel(_found(f"ك{FATHA}ت{FATHA}ب{FATHA}", f"ك{FATHA}ت{FATHA}ب{FATHA}"))
    assert {index for _, index in found} == {0, 2}


def test_a_case_ending_before_a_realized_tanween_is_not_mid_word():
    # عِلمُن: the damma on م is the case ending; the ن after it realizes the tanween.
    reference = f"ع{KASRA}لم{DAMMA}ن"
    with_tanween = _tashkeel(_found(reference, reference, tanween=[True]))
    without = _tashkeel(_found(reference, reference, tanween=[False]))
    assert (T("damma", "matched"), 3) not in with_tanween
    assert (T("damma", "matched"), 3) in without
    assert (T("sukun", "empty"), 2) in with_tanween  # the ل is a letter of the word


def test_word_spans_come_from_the_offsets_not_the_spaces():
    # Two words joined with no space (as wasl merges leave them): the fatha ending word 0
    # is word-final, though a consonant follows it in the string.
    reference = f"ك{FATHA}ت{FATHA}ب{FATHA}ر"
    split = _tashkeel(_found(reference, reference, offsets=[0, 4, len(reference)]))
    joined = _tashkeel(_found(reference, reference))
    assert {index for _, index in split} == {0, 4}
    assert {index for _, index in joined} == {0, 2, 4}


def test_a_mid_word_sukun_is_stratified_by_whether_the_base_voweled_it():
    reference = f"ي{FATHA}علم{DAMMA}"  # يَعلَمُ without the fatha: ع and ل are sakin
    assert (T("sukun", "empty"), 2) in _found(reference, reference)
    assert (T("sukun", "haraka"), 2) in _found(reference, f"ي{FATHA}ع{FATHA}لم{DAMMA}")
    assert (T("sukun", "empty", "other"), 2) in _found(reference, reference, h448="ي")


def test_a_qalqala_letter_is_a_sukun_site_only_mid_word():
    mid = f"ي{FATHA}قڇت{DAMMA}ل"
    end = f"ي{FATHA}قڇ"
    assert (T("sukun", "empty"), 2) in _found(mid, mid)
    assert not [s for s in _found(end, end) if s[0].startswith("new_audit:sukun")]


def test_the_halves_of_a_geminate_are_not_sukun_sites():
    reference = f"ر{FATHA}بب{KASRA}ي"
    assert not [s for s in _found(reference, reference) if s[0].startswith("new_audit:sukun")]


def test_a_geminate_is_stratified_by_whether_the_base_decoded_it_single():
    reference = f"ر{FATHA}بب{KASRA}ي{FATHA}"
    assert ("new_audit:shaddah:held:base_rest", 2) in _found(reference, reference)
    single = f"ر{FATHA}ب{KASRA}ي{FATHA}"
    assert ("new_audit:shaddah:held:base_single", 2) in _found(reference, single)


def test_a_geminate_across_a_word_boundary_is_its_own_stratum():
    # تُ|وَ: a tanween assimilated into the next word's و (#86: usually a missed pause).
    reference = f"ك{DAMMA}تُوو{FATHA}ل"
    offsets = [0, 5, len(reference)]  # the first و ends word 0, the second starts word 1
    across = _found(reference, reference, offsets=offsets)
    assert ("new_audit:shaddah:held:cross_word", 4) in across
    assert not {f for f in across if f[0].startswith("new_audit:shaddah:held:base_")}
    within = _found(reference, reference, offsets=[0, len(reference)])
    assert ("new_audit:shaddah:held:base_rest", 4) in within


def test_every_single_consonant_is_a_not_held_site_stratified_by_a_base_doubling():
    reference = f"ر{FATHA}ب{KASRA}ي{FATHA}"
    doubled = _found(reference, f"ر{FATHA}بب{KASRA}ي{FATHA}")
    assert ("new_audit:shaddah:not_held:base_double", 2) in doubled
    assert {("new_audit:shaddah:not_held:base_rest", i) for i in (0, 4)} <= doubled
    assert ("new_audit:shaddah:not_held:base_rest", 2) in _found(reference, reference)


@pytest.mark.parametrize("pair, said, heard", [("ذ↔ز", "ذ", "ز"), ("ذ↔ظ", "ظ", "ذ")])
def test_pair_sites_are_directional_and_include_dhal_zah(pair, said, heard):
    reference = f"ك{FATHA}{said}{FATHA}ب"
    decode = f"ك{FATHA}{heard}{FATHA}ب"
    assert (f"new_audit:{said}→{heard}:base_partner", 2) in _found(reference, decode)
    assert (f"new_audit:{said}→{heard}:base_rest", 2) in _found(reference, reference)


def test_eligibility_never_depends_on_the_decode():
    reference = f"ي{FATHA}ذك{DAMMA}رر{FATHA}قڇت{DAMMA}ل"
    decodes = [reference, "", f"ي{KASRA}زكررقتل", f"ي{FATHA}ذذ{FATHA}ك{DAMMA}ر{FATHA}",
               f"ي{FATHA}ظك{DAMMA}ر{FATHA}قتت{DAMMA}ل"]
    sites = [
        {(f.mark, f.prescribed, f.reference_index)
         for f in segment_sites(reference, base, h448, [0, len(reference)], [False])}
        for base, h448 in zip(decodes, decodes[::-1])
    ]
    assert all(found == sites[0] for found in sites)


def test_word_offsets_must_partition_the_reference():
    with pytest.raises(ValueError, match="partition"):
        word_limits("كتب", [0, 2], [False])


def test_haraka_and_sukun_share_one_question_and_one_id_suffix():
    assert mode_of("fatha") == mode_of("sukun") == "tashkeel"
    assert mode_of("shaddah") == "shaddah" and mode_of("ذ↔ظ") == "consonant"
    assert site_id("c.wav", 1, 4, "kasra") == site_id("c.wav", 1, 4, "sukun") \
        == "new_audit:c.wav#1@4:tashkeel"


# --- excerpts --------------------------------------------------------------------------


def _segment(**overrides) -> PoolSegment:
    fields = dict(segment_index=0, word_start=2, word_end=6, start_sample=16000,
                  end_sample=160000, reference="aa bb cc dd", raw_word_offsets=(0, 3, 6, 9, 11),
                  kept=True)
    return PoolSegment(**{**fields, **overrides})


WORD_TIMES = (0.0, 0.5, 1.0, 3.0, 5.0, 7.0, 9.0, 10.0)


CLIP_SAMPLES = 200000


def test_the_excerpt_is_the_carrier_word_and_one_either_side_padded():
    # Reference index 4 is in local word 1, i.e. word 3, so words 2-4 play: word 2's onset
    # (1.0 s) to word 5's (7.0 s), padded 0.3 s each side, past the segment's 1.0 s start.
    assert excerpt_span(_segment(), WORD_TIMES, 0, 4, CLIP_SAMPLES) == (11200, 116800)
    # Index 10 is in word 5, the segment's last: words 4-5, 5.0 s to 9.0 s, padded.
    assert excerpt_span(_segment(), WORD_TIMES, 0, 10, CLIP_SAMPLES) == (75200, 148800)


def test_the_excerpt_is_clamped_only_to_the_clip():
    assert excerpt_span(_segment(), WORD_TIMES, 0, 4, 115000) == (11200, 115000)


def test_the_whole_segment_plays_when_the_word_times_cannot_place_the_words():
    whole = (16000, 160000)
    assert excerpt_span(_segment(), WORD_TIMES, 1, 4, CLIP_SAMPLES) == whole  # a re-read
    assert excerpt_span(_segment(), (), 0, 4, CLIP_SAMPLES) == whole
    late = _segment(start_sample=150000, end_sample=160000)  # words end before it starts
    assert excerpt_span(late, WORD_TIMES, 0, 4, CLIP_SAMPLES) == (150000, 160000)
    squeezed = (0.0, 0.5, 1.0, 1.1, 1.2, 1.3, 1.4, 1.5)
    tiny = _segment(start_sample=17600, end_sample=18000)
    assert excerpt_span(tiny, squeezed, 0, 4, CLIP_SAMPLES) == (17600, 18000)


# --- provenance ------------------------------------------------------------------------


def test_pool_sites_take_their_provenance_from_the_registry_and_their_item_from_the_segment():
    reference = f"ي{FATHA}علم{DAMMA}"
    kept = _segment(reference=reference, raw_word_offsets=(0, len(reference)), word_end=3)
    dropped = replace(kept, segment_index=1, kept=False)
    clip = PoolClip("c.wav", ("uniform",), 0.25, "2:2", 3, 4, None, 0, 4, 0.0, 10.0,
                    WORD_TIMES[:5], (kept, dropped))
    registry = {"c.wav": StagedClip("c.wav", 77, 5, 9, "2:2", 200000, "b" * 64,
                                    ("mining_pool",))}
    candidates = pool_candidates([clip], {"c.wav#0": reference}, {"c.wav#0": reference},
                                 registry, lambda surah_ayah: [False] * 4)
    assert candidates and all(c.clip_inclusion_probability == 0.25 for c in candidates)
    sukun = next(c for c in candidates if c.site.mark == "sukun")
    site = sukun.site
    assert (site.shard, site.audio_sha256, site.start_sample, site.end_sample) == (
        77, "b" * 64, 16000, 160000)
    assert (site.site_id, site.heard, site.stratum) == (
        "new_audit:c.wav#0@2:tashkeel", PENDING, T("sukun", "empty"))
    assert (sukun.word_start, sukun.word_offsets) == (2, (0, len(reference)))


def test_the_excerpt_never_depends_on_what_a_decode_emitted_at_the_carrier():
    reference = f"ي{FATHA}علم{DAMMA}"
    segment = _segment(reference=reference, raw_word_offsets=(0, len(reference)), word_end=3)
    clip = PoolClip("c.wav", ("uniform",), 1.0, "2:2", 3, 4, None, 0, 4, 0.0, 10.0,
                    WORD_TIMES[:5], (segment,))
    registry = {"c.wav": StagedClip("c.wav", 77, 5, 9, "2:2", 200000, "b" * 64,
                                    ("mining_pool",))}
    voweled = f"ي{FATHA}ع{KASRA}لم{DAMMA}"

    def sukun_site(base, h448):
        (candidate,) = [c for c in pool_candidates([clip], {"c.wav#0": base}, {"c.wav#0": h448},
                                                   registry, lambda _: [False] * 4)
                        if c.site.reference_index == 2 and c.site.mark == "sukun"]
        return candidate

    plain, other = sukun_site(reference, reference), sukun_site(voweled, voweled)
    assert plain.site.stratum != other.site.stratum
    assert plain.excerpt == other.excerpt


def test_only_pending_p35_sites_are_re_adjudicated_on_their_own_segment():
    segment = _segment(reference=f"ك{FATHA}ت{FATHA}ب", raw_word_offsets=(0, 5), word_end=3)
    pending = _site(1, "p35_fixture:ذ↔ز", source="p35_fixture", start_sample=16000,
                    end_sample=160000, stratum_population=30)
    heard = replace(pending, site_id="other", heard="fatha")
    segments = {("clip1.wav", 16000): (segment, (), 0, 200000)}
    (candidate,) = p35_candidates([pending, heard], segments)
    assert candidate.site is pending and candidate.excerpt == (16000, 160000)
    moved = replace(pending, end_sample=150000)
    with pytest.raises(ValueError, match="no longer matches"):
        p35_candidates([moved], segments)


# --- the draw --------------------------------------------------------------------------


FATHA_EMPTY = T("fatha", "empty")
SUKUN_HARAKA = T("sukun", "haraka")


def _site(index: int, stratum: str = FATHA_EMPTY, **overrides) -> TruthSite:
    fields = dict(
        site_id=f"{stratum}#{index}", source="new_audit",
        assumes_competent_reciter=False, audio_filename=f"clip{index}.wav", shard=40,
        start_sample=0, end_sample=32000, audio_sha256="a" * 64, surah_ayah="2:2",
        reference=f"ك{FATHA}ت{FATHA}ب", reference_index=2, mark="fatha", prescribed="fatha",
        heard=PENDING, stratum=stratum, stratum_population=0,
    )
    return TruthSite(**{**fields, **overrides})


def _candidates(n: int, stratum: str = FATHA_EMPTY, clip_p: float = 0.5):
    return [Candidate(_site(i, stratum), clip_p, 0, (0, 5), (100, 16100)) for i in range(n)]


def test_each_stratum_draws_its_size_with_its_population_and_probabilities():
    rows = draw(_candidates(40) + _candidates(3, SUKUN_HARAKA),
                {FATHA_EMPTY: 10, SUKUN_HARAKA: 5})
    fatha = [r for r in rows if r.site.stratum == FATHA_EMPTY]
    sukun = [r for r in rows if r.site.stratum == SUKUN_HARAKA]
    assert len(fatha) == 10 and len(sukun) == 3
    assert {r.site.stratum_population for r in fatha} == {40}
    assert {(r.draw_probability, r.inclusion_probability) for r in fatha} == {(0.25, 0.125)}
    assert {r.draw_probability for r in sukun} == {1.0}


def test_a_stratum_without_a_size_is_counted_but_not_drawn():
    assert draw(_candidates(5), {}) == []


def test_sizes_must_name_known_strata():
    with pytest.raises(ValueError, match="unknown strata"):
        draw(_candidates(5), {"new_audit:fatha:base_empty": 2})
    assert set(DEFAULT_SIZES) <= set(STRATA)


def test_the_draw_survives_re_mining_a_changed_population():
    before = {r.site.site_id for r in draw(_candidates(200), {FATHA_EMPTY: 20})}
    grown = _candidates(210)[5:]  # five sites lost, fifteen found
    after = {r.site.site_id for r in draw(grown, {FATHA_EMPTY: 20})}
    assert len(before & after) >= 17


def test_the_draw_ignores_candidate_order():
    candidates = _candidates(50)
    sizes = {FATHA_EMPTY: 7}
    assert draw(candidates, sizes) == draw(candidates[::-1], sizes)


def test_census_rows_are_certain_and_keep_their_own_population():
    site = _site(1, "p35_fixture:shadda", stratum_population=24)
    (row,) = census([Candidate(site, 1.0, 0, (0, 5), (0, 32000))])
    assert (row.draw_probability, row.inclusion_probability) == (1.0, 1.0)
    assert row.site.stratum_population == 24


def test_the_queue_is_shuffled_across_strata_deterministically():
    rows = draw(_candidates(30) + _candidates(30, T("sukun", "empty")),
                {FATHA_EMPTY: 30, T("sukun", "empty"): 30})
    queue = shuffled(rows)
    assert queue == shuffled(rows[::-1])
    strata = [r.site.stratum for r in queue]
    assert strata != sorted(strata) and strata != sorted(strata, reverse=True)


def test_the_summary_lists_every_stratum_and_the_listening_time():
    registry = {f"clip{i}.wav": StagedClip(f"clip{i}.wav", 40, i, i % 3, "2:2", 32000,
                                           "a" * 64, ("mining_pool",)) for i in range(40)}
    candidates = _candidates(40)
    rows = draw(candidates, {FATHA_EMPTY: 4})
    summary = summarize([c.site for c in candidates], rows, registry)
    assert list(summary["strata"])[: len(STRATA)] == list(STRATA)
    cell = summary["strata"][FATHA_EMPTY]
    assert (cell["population"], cell["population_reciters"], cell["sampled"]) == (40, 3, 4)
    # 1 s excerpts, played twice, plus 4 s to answer: 6 s a site.
    assert summary["listening_minutes"] == round(4 * 6 / 60, 1)
    assert summary["strata"][T("kasra", "empty")]["population"] == 0


# --- the worklist file -----------------------------------------------------------------


def _rows():
    return draw(_candidates(4), {FATHA_EMPTY: 4})


def test_the_worklist_round_trips(tmp_path):
    path = tmp_path / "worklist.jsonl"
    write_worklist(_rows(), path)
    assert load_worklist(path) == _rows()


@pytest.mark.parametrize(
    "change, message",
    [
        (lambda r: replace(r, site=replace(r.site, heard="fatha")), "skeleton"),
        (lambda r: replace(r, inclusion_probability=0.9), "clip x draw"),
        (lambda r: replace(r, excerpt_start_sample=-1), "span of samples"),
        (lambda r: replace(r, word_offsets=(0, 4)), "word_offsets"),
        (lambda r: replace(r, draw_probability=1), "floats"),
        (lambda r: replace(r, site=replace(r.site, reference_index=3)), "carry"),
    ],
)
def test_a_malformed_row_is_never_written(tmp_path, change, message):
    rows = _rows()
    with pytest.raises(ValueError, match=message):
        write_worklist([change(rows[0]), *rows[1:]], tmp_path / "worklist.jsonl")
    assert not (tmp_path / "worklist.jsonl").exists()


# --- verdicts --------------------------------------------------------------------------


def test_verdicts_round_trip_sorted_by_site_id(tmp_path):
    path = tmp_path / "verdicts.jsonl"
    verdicts = {v.site_id: v for v in [Verdict("b", "sukun", "late"), Verdict("a", "held")]}
    write_verdicts(verdicts, path)
    assert read_verdicts(path) == verdicts
    assert [json.loads(line)["site_id"] for line in path.read_text().splitlines()] == ["a", "b"]


def test_a_missing_or_empty_verdict_file_is_an_unstarted_session(tmp_path):
    assert read_verdicts(tmp_path / "absent.jsonl") == {}
    (tmp_path / "empty.jsonl").write_text("")
    assert read_verdicts(tmp_path / "empty.jsonl") == {}


def test_a_naturalness_verdict_round_trips_beside_what_was_said(tmp_path):
    path = tmp_path / "verdicts.jsonl"
    verdicts = {"a": Verdict("a", "held"), "e": Verdict("e", "not_held", "", "unnatural")}
    write_verdicts(verdicts, path)
    assert read_verdicts(path) == verdicts
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    assert lines == [{"site_id": "a", "heard": "held", "note": ""},
                     {"site_id": "e", "heard": "not_held", "note": "", "natural": "unnatural"}]


def test_a_verdicts_file_without_naturalness_reads_and_rewrites_byte_for_byte(tmp_path):
    # The schema change is additive: files written before it (and the tracked one) load
    # unchanged, and a rewrite reproduces them exactly.
    old = ('{"heard": "fatha", "note": "late", "site_id": "a"}\n'
           '{"heard": "held", "note": "", "site_id": "b"}\n')
    path = tmp_path / "verdicts.jsonl"
    path.write_text(old, encoding="utf-8")
    verdicts = read_verdicts(path)
    assert all(v.natural is None for v in verdicts.values())
    write_verdicts(verdicts, path)
    assert path.read_text(encoding="utf-8") == old
    read_verdicts(VERDICTS_PATH)


@pytest.mark.parametrize("text", [
    '{"site_id": "a", "heard": "held", "note": "", "natural": "sounds fine"}\n',
    '{"site_id": "a", "heard": "held", "note": "", "natural": null}\n',
    '{"site_id": "a", "heard": "held"}\n',
    '{"site_id": "a", "heard": "held", "note": ""}\n'
    '{"site_id": "a", "heard": "not_held", "note": ""}\n',
])
def test_a_malformed_verdict_file_is_refused(tmp_path, text):
    (tmp_path / "v.jsonl").write_text(text)
    with pytest.raises(ValueError):
        read_verdicts(tmp_path / "v.jsonl")


def test_an_interrupted_save_keeps_the_verdicts_already_recorded(tmp_path):
    path = tmp_path / "verdicts.jsonl"
    write_verdicts({"a": Verdict("a", "held")}, path)
    original = path.read_bytes()

    class Exploding(dict):
        def __getitem__(self, key):
            raise RuntimeError("crash mid-write")

    with pytest.raises(RuntimeError):
        write_verdicts(Exploding({"x": None}), path)
    assert path.read_bytes() == original


def test_a_verdict_fills_its_pending_site_by_id_and_nothing_else():
    from training.test_site_outcomes import DHAKARA, DHAL, ZAI, make

    pending = make(DHAKARA, 2, "ذ↔ز", DHAL, PENDING, site_id="p35:a")
    accepted = make(DHAKARA, 2, "ذ↔ز", DHAL, DHAL, site_id="p35:b")
    untouched = make(DHAKARA, 2, "ذ↔ز", DHAL, PENDING, site_id="p35:c")
    verdicts = {
        "p35:a": Verdict("p35:a", ZAI),
        "p35:b": Verdict("p35:b", DHAL),  # agrees with the file
        "new_audit:x": Verdict("new_audit:x", "fatha"),  # another file's site
    }
    result = adjudicated([pending, accepted, untouched], verdicts)
    assert [s.heard for s in result] == [ZAI, DHAL, PENDING]
    with pytest.raises(ValueError, match="in its file"):
        adjudicated([accepted], {"p35:b": Verdict("p35:b", ZAI)})
    with pytest.raises(ValueError, match="not an answer"):
        adjudicated([pending], {"p35:a": Verdict("p35:a", "held")})


def test_every_mark_is_answerable_and_pending_is_never_an_answer():
    assert hearable("kasra") == {"fatha", "damma", "kasra", "sukun", "unclear"}
    assert hearable("shaddah") == {"held", "not_held", "unclear"}
    assert hearable("ذ↔ظ") == {"ذ", "ظ", "unclear"}


# --- the committed worklist ------------------------------------------------------------


def test_the_committed_worklist_loads_and_agrees_with_its_summary_and_the_registry():
    rows = load_worklist(WORKLIST_PATH)
    summary = json.loads(SUMMARY_PATH.read_text(encoding="utf-8"))
    registry = load_staged_clips()
    assert len(rows) == summary["sites"]
    for stratum, cell in summary["strata"].items():
        drawn = [r for r in rows if r.site.stratum == stratum]
        assert len(drawn) == cell["sampled"], stratum
        assert all(r.site.stratum_population == cell["population"] for r in drawn), stratum
    for row in rows:
        clip = registry[row.site.audio_filename]
        assert (row.site.shard, row.site.audio_sha256) == (clip.shard, clip.audio_sha256)
        assert row.site.end_sample <= clip.num_samples
    pending_p35 = [r for r in rows if r.site.source == "p35_fixture"]
    assert len(pending_p35) == 23 and all(r.inclusion_probability == 1.0 for r in pending_p35)
    assert all(isinstance(r, SessionSite) for r in rows)


# --- the session's audio ---------------------------------------------------------------


def test_one_rsync_list_covers_both_remote_directories():
    root, files = fetch_list({"/root/scratch/issue-83/stage/clips": {"b.wav", "a.wav"},
                              "/root/scratch/issue-88/stage/edits/audio": {"se_1.wav"}})
    assert root == "/root/scratch"
    assert files == ["issue-83/stage/clips/a.wav", "issue-83/stage/clips/b.wav",
                     "issue-88/stage/edits/audio/se_1.wav"]
    with pytest.raises(ValueError, match="one name"):
        fetch_list({"/x/a": {"c.wav"}, "/x/b": {"c.wav"}})

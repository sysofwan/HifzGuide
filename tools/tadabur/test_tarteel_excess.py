"""Tests for tadabur.tarteel_excess: the frame, the draw, the staged audio and the blind page (#160).

Harness-free and offline: grades, clips and the dataset's rows are synthetic.
"""

from __future__ import annotations

import io
import json

import numpy as np
import pytest
import soundfile as sf

from tadabur import tarteel_excess as te
from tadabur.nonpro_sources import audio_sha256
from tadabur.qud_text import ItemReference
from tadabur.tarteel_split import CONFIRM, LOOK, half
from tadabur.tashkeel_audit_ui import blind_reference
from training.external_corpus import WordGrade
from training.nonpro_eval import ClipItem
from training.word_flags import Flag

# Three canonical words; Muraja joins the last two into its second phoneme word.
REFERENCE = "قَالَ سَططَ نَا"
OFFSETS = (0, 6, 12, 15)
LOOK_CLIPS = ("0_1_0.wav", "2_1_2.wav", "4_1_4.wav", "5_1_5.wav", "6_1_6.wav", "9_1_9.wav")
CONFIRM_CLIP = "1_1_1.wav"
OC = "other_consonant"
SAMPLES = 1600


def _item(clip: str) -> ClipItem:
    return ClipItem(f"{clip}#0", "tarteel_v1", clip, 1, 1, 0, SAMPLES, "e" * 64,
                    ItemReference(("a", "b", "c"), (0, 1, 2), REFERENCE, OFFSETS), "", 1)


def _flag(*substitutions: str) -> Flag:
    consonants = tuple((OC, s) for s in substitutions)
    return Flag((OC,) if consonants else ("tashkeel",), (), "minor", ("g0:x",),
                consonants=consonants)


def _grades(clip: str, h448: tuple, base: tuple) -> list[WordGrade]:
    """Both Muraja words of ``clip`` under each model: a flag (or ``None``) per word."""
    item = _item(clip)
    return [WordGrade(model, "shipped_default", item, k, 2, k + 1, (0, 1)[k],
                      "flagged" if flag else "not_flagged", flag)
            for model, flags in (("h448", h448), ("base", base)) for k, flag in enumerate(flags)]


RANKS = {(OC, "م→∅"): 0, (OC, "ز→س"): 1, (OC, "ل→ن"): 2}


def test_the_clips_used_here_sit_in_the_halves_they_claim():
    assert {half(c) for c in LOOK_CLIPS} == {LOOK} and half(CONFIRM_CLIP) == CONFIRM


def test_the_frame_is_the_look_half_s_h448_flags_the_teacher_grades_correct():
    grades = (_grades(LOOK_CLIPS[0], (None, _flag("ل→ن")), (None, None))
              + _grades(LOOK_CLIPS[1], (_flag("ل→ن"), None), (_flag("ل→ن"), None))
              + _grades(CONFIRM_CLIP, (None, _flag("ل→ن")), (None, None)))
    [word] = te.excess_words(grades, RANKS)
    assert word.clip == LOOK_CLIPS[0] and word.stratum == "ل→ن"
    # Muraja's second word covers the second and third canonical words.
    assert (word.position_from, word.position_to) == (1, 3)


def test_the_top_substitution_by_the_tally_picks_the_stratum():
    assert te.stratum_of([(OC, "ل→ن"), (OC, "ز→س")], RANKS) == "ز→س"
    assert te.stratum_of([(OC, "ز→س"), (OC, "م→∅")], RANKS) == te.OTHER  # م→∅ is not named
    assert te.stratum_of([(OC, "ن→∅")], RANKS) == "ن→∅"  # named, though this tally lacks it
    assert te.stratum_of([("shaddah", "ن→∅"), (OC, "ل→ن")], RANKS) == "ل→ن"
    assert te.stratum_of([], RANKS) == te.OTHER


def test_the_issue_s_strata_are_the_tally_s_order():
    ranks = te.substitution_ranks(json.loads(te.NONPRO_REPORT.read_text(encoding="utf-8")))
    assert sorted(te.NAMED, key=lambda s: ranks[(OC, s)]) == list(te.NAMED)


def _frame() -> list[te.ExcessWord]:
    """Clip 0 has two excess words (one per stratum), the others one each."""
    words = []
    for k, clip in enumerate(LOOK_CLIPS):
        for word, sub in ([(0, "ل→ن"), (1, "ز→س")] if k == 0 else [(k % 2, ("ل→ن", "ز→س")[k % 2])]):
            grade = WordGrade("h448", "shipped_default", _item(clip), word, 2, word + 1,
                              (0, 1)[word], "flagged", _flag(sub))
            words.append(te.ExcessWord(grade, word, (1, 3)[word], sub))
    return words


def test_the_draw_keeps_one_word_per_clip_and_records_both_stages():
    frame = _frame()
    sizes = {**{s: 1 for s in te.NAMED}, te.OTHER: 1}
    rows = te.draw(frame, sizes)
    assert rows == te.draw(list(reversed(frame)), sizes)  # a function of the ids alone
    assert len({r.audio_filename for r in rows}) == len(rows) == 2
    for row in rows:
        members = [w for w in frame if w.stratum == row.stratum]
        assert row.stratum_words == len(members)
        assert row.clip_words == sum(w.clip == row.audio_filename for w in frame)
        assert row.representative_probability == 1 / row.clip_words
        assert row.draw_probability == 1 / row.stratum_population
        assert row.inclusion_probability == pytest.approx(
            row.representative_probability * row.draw_probability)
    # The representatives partition the clips: six clips, one each.
    assert sum(r.stratum_population for r in rows) == len(LOOK_CLIPS)
    everything = te.draw(frame)  # the default sizes take every representative
    assert {r.draw_probability for r in everything} == {1.0}


def test_a_row_round_trips_and_the_page_highlights_the_whole_muraja_word():
    row = next(r for r in te.draw(_frame()) if r.position_from == 1)
    assert te.parse_row(te.row_json(row), "here") == row
    site = row.word_site()
    assert blind_reference(site, {}) == ("قَالَ ", "سَططَ نَا", "")
    assert (site.excerpt_start_sample, site.excerpt_end_sample) == (0, SAMPLES)


def _wav(seed: int) -> bytes:
    buffer = io.BytesIO()
    noise = np.random.default_rng(seed).uniform(-0.5, 0.5, SAMPLES).astype(np.float32)
    sf.write(buffer, noise, 16000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _published(rows) -> dict[str, bytes]:
    return {r.audio_filename: _wav(k) for k, r in enumerate(rows)}


def _with_checksums(rows, published):
    from dataclasses import replace

    return [replace(r, audio_sha256=audio_sha256(published[r.audio_filename])) for r in rows]


def _stage(rows, published, audio_dir, recorded=None):
    def locate(dataset, names):
        return {name: ("data/train-00000-of-00008.parquet", k % 2)
                for k, name in enumerate(sorted(names))}

    def read_group(dataset, shard, group):
        return [{"audio": {"path": n, "bytes": b}} for n, b in published.items()]

    return te.stage(rows, audio_dir, recorded, locate, read_group)


def test_staging_checks_the_dataset_s_bytes_and_length_against_156(tmp_path):
    rows = te.draw(_frame())
    published = _published(rows)
    good = _with_checksums(rows, published)
    staged = _stage(good, published, tmp_path / "a")
    te.verify_audio(good, staged, tmp_path / "a")
    assert _stage(good, published, tmp_path / "b", staged) == staged  # re-staged alike
    with pytest.raises(ValueError, match="SHA-256"):
        _stage(rows, published, tmp_path / "c")  # #156's checksum is "e" * 64
    from dataclasses import replace

    with pytest.raises(ValueError, match="samples"):
        _stage([replace(r, end_sample=SAMPLES - 1) for r in good], published, tmp_path / "d")
    name = good[0].audio_filename
    wrong = {**staged, name: replace(staged[name], staged_sha256="0" * 64)}
    with pytest.raises(ValueError, match="records"):
        _stage(good, published, tmp_path / "e", wrong)
    assert not (tmp_path / "e" / name).exists()
    with pytest.raises(ValueError, match="not the staged clip"):
        te.verify_audio(good, wrong, tmp_path / "a")


def _session_dir(tmp_path):
    rows = te.draw(_frame())
    published = _published(rows)
    rows = _with_checksums(rows, published)
    directory, audio_dir = tmp_path / "session", tmp_path / "audio"
    te.write_worklist(rows, directory / "worklist.jsonl")
    (directory / "worklist.summary.json").write_text(json.dumps(
        {"frozen": {"worklist.jsonl": te.sha256_of(directory / "worklist.jsonl")}}))
    te.write_staged(_stage(rows, published, audio_dir), directory / "staged.jsonl")
    return rows, directory, audio_dir


def test_the_page_asks_the_word_question_blind(tmp_path):
    rows, directory, audio_dir = _session_dir(tmp_path)
    state = te.session(audio_dir, directory)
    payload = state.payload()
    assert payload["progress"] == {"answered": 0, "total": len(rows)}
    assert {s["mode"] for s in payload["sites"]} == {"word"}
    assert {s["carrier"] for s in payload["sites"]} <= {"قَالَ ", "سَططَ نَا"}
    shown = json.dumps(payload, ensure_ascii=False)
    for secret in ("h448", "stratum", "→", "probability", *(r.site_id for r in rows)):
        assert secret not in shown
    key = payload["sites"][0]["key"]
    state.record(key, "mistake", "dropped the hamza", None)
    saved = (directory / "verdicts.jsonl").read_text(encoding="utf-8")
    assert '"heard": "mistake"' in saved
    assert len(state.audio(key, whole=False)) > 2 * SAMPLES  # the whole clip, as 16-bit PCM


def test_a_worklist_the_summary_did_not_freeze_is_refused(tmp_path):
    _, directory, audio_dir = _session_dir(tmp_path)
    path = directory / "worklist.jsonl"
    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="froze"):
        te.session(audio_dir, directory)

"""Tests for the whole-clip audit view builder (``training.whole_clip_audit``, #26).

Pure logic over the same scored segment manifest + per-clip status sidecar #25 consumes —
no GPU, no quran-transcript. Verifies the builder reconstructs the whole-clip concatenated
label + per-segment breakdown for an eligible clip, projects its exact training windows, and
surfaces every excluded clip with the canonical exclusion reason (so the #6 auditor sees the
real training data path, not a drifting second opinion).
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from tadabur.clip_status import ClipStatus, write_clip_status
from training.windowed_labels import (
    EXCLUDE_DROPPED_SEGMENT,
    EXCLUDE_SEGMENT_CROSSES_WINDOW,
)
from training.whole_clip_audit import build_whole_clip_audit


def _raw(ref):
    """A tashkeel-bearing counterpart of ``ref`` — a fatha after its first phoneme.

    The audit reads the *raw* reference (ADR-0003), so making it differ from the
    vowel-stripped ``reference_phonemes`` keeps these tests honest about which one is used.
    """
    return ref[0] + "\u064e" + ref[1:]


def _row(clip, index, w0, w1, s0, s1, ref, uthmani, reciter=1, surah="78:2"):
    """One scored segment-manifest row, as ``tadabur.segment_score`` emits it."""
    raw = _raw(ref)
    n_words = w1 - w0
    # Any partition of ``raw`` into ``n_words`` pieces will do: these tests assert whole-segment
    # labels, so only the endpoints of the offset list are load-bearing.
    bounds = [round(i * len(raw) / n_words) for i in range(n_words)] + [len(raw)]
    return {
        "raw_reference_phonemes": raw,
        "raw_word_offsets": bounds,
        "audio_filename": f"{clip}__seg{index}",
        "clip_audio_filename": clip,
        "surah_ayah": surah,
        "reciter_id": reciter,
        "segment_index": index,
        "word_start": w0,
        "word_end": w1,
        "start_s": s0,
        "end_s": s1,
        "reference_phonemes": ref,
        "uthmani": uthmani,
    }


def _status(clip, n_words, segs, reciter=1, surah="78:2", skip=None):
    return ClipStatus(
        audio_filename=clip,
        surah_ayah=surah,
        reciter_id=reciter,
        n_words=n_words,
        duration_s=segs[-1]["end_s"] if segs else 4.0,
        recitation_start_s=segs[0]["start_s"] if segs else 0.0,
        recitation_end_s=segs[-1]["end_s"] if segs else 4.0,
        skip_reason=skip,
    )


def _write(tmp_path, rows, statuses):
    manifest = tmp_path / "segments.jsonl"
    with open(manifest, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    status_path = tmp_path / "clip_status.jsonl"
    write_clip_status(status_path, statuses)
    return manifest, status_path


def test_included_clip_reconstructs_label_breakdown_and_windows(tmp_path):
    # Three short segments inside one 5 s window: one training window owns all three, and the
    # whole-clip label is their realized references concatenated in word order.
    rows = [
        _row("a.wav", 0, 0, 1, 0.0, 1.5, "ءا", "أ"),
        _row("a.wav", 1, 1, 2, 1.5, 3.0, "بب", "ب"),
        _row("a.wav", 2, 2, 4, 3.0, 4.4, "تت", "ت ث"),
    ]
    manifest, status_path = _write(tmp_path, rows, [_status("a.wav", 4, rows)])

    audit = build_whole_clip_audit(manifest, status_path)
    assert audit.clips_included == 1 and audit.clips_excluded == 0

    view = audit.views[0]
    assert view.clip_id == "a.wav"
    assert view.included is True and view.exclusion_reason is None
    assert view.whole_clip_label == "ءَابَبتَت"  # tashkeel survives (ADR-0003)
    # Per-segment breakdown carries word ranges + Uthmani words + realized reference, ordered.
    assert [(s.segment_index, s.word_start, s.word_end, s.uthmani, s.reference)
            for s in view.segments] == [
        (0, 0, 1, "أ", _raw("ءا")), (1, 1, 2, "ب", _raw("بب")),
        (2, 2, 4, "ت ث", _raw("تت"))]
    # The exact training unit: one window whose CTC target is the concatenation.
    assert len(view.windows) == 1
    win = view.windows[0]
    assert win.phoneme_label == "ءَابَبتَت"
    assert win.segment_indices == (0, 1, 2)
    assert (win.word_start, win.word_end) == (0, 4)
    assert len(win.phoneme_label) < win.logit_frames


def test_dropped_segment_clip_is_excluded_with_reason(tmp_path):
    # A dropped middle segment leaves a word-coverage gap (words 1-2 unaccounted): the whole
    # clip is excluded from training and surfaced with the canonical reason.
    rows = [
        _row("b.wav", 0, 0, 1, 0.0, 1.5, "ءا", "أ"),
        _row("b.wav", 2, 2, 4, 3.0, 4.4, "تت", "ت ث"),
    ]
    manifest, status_path = _write(tmp_path, rows, [_status("b.wav", 4, rows)])

    audit = build_whole_clip_audit(manifest, status_path)
    view = audit.views[0]
    assert view.included is False
    assert view.exclusion_reason == EXCLUDE_DROPPED_SEGMENT
    assert view.windows == ()
    # The auditor still sees the surviving segments that could not be assembled.
    assert [s.segment_index for s in view.segments] == [0, 2]
    assert audit.exclusions_by_reason == {EXCLUDE_DROPPED_SEGMENT: 1}


def test_skip_reason_clip_with_no_segments_is_surfaced(tmp_path):
    # A phonetizer-unsupported clip leaves no manifest rows at all — only a skip status.
    manifest, status_path = _write(
        tmp_path, [], [_status("c.wav", 3, [], skip="phonetizer_unsupported")]
    )
    audit = build_whole_clip_audit(manifest, status_path)
    view = audit.views[0]
    assert view.included is False
    assert view.exclusion_reason == "phonetizer_unsupported"
    assert view.segments == () and view.whole_clip_label == ""


def test_views_are_ordered_and_mixed_summary_is_deterministic(tmp_path):
    good = [_row("a.wav", 0, 0, 3, 0.0, 4.0, "ءبت", "أ ب ت")]
    crossing = [  # segment 1 straddles the 5 s window edge → segment_crosses_window
        _row("z.wav", 0, 0, 2, 0.0, 3.0, "ءبت", "أ ب"),
        _row("z.wav", 1, 2, 4, 3.0, 6.0, "جحخ", "ج ح"),
    ]
    rows = good + crossing
    statuses = [_status("a.wav", 3, good), _status("z.wav", 4, crossing)]
    manifest, status_path = _write(tmp_path, rows, statuses)

    audit = build_whole_clip_audit(manifest, status_path)
    assert [v.clip_id for v in audit.views] == ["a.wav", "z.wav"]  # sorted by clip id
    assert audit.clips_included == 1 and audit.clips_excluded == 1
    assert audit.exclusions_by_reason == {EXCLUDE_SEGMENT_CROSSES_WINDOW: 1}
    # asdict round-trips cleanly for the JSON API (tuples serialize as arrays).
    payload = json.dumps([asdict(v) for v in audit.views], ensure_ascii=False)
    assert "phoneme_label" in payload

"""The catch-loss audit through the #61 listening UI (#225): a tlog item source for its word question.

``SessionState.load_catch_loss(worklist, verdicts, clip_cache, uthmani_words, review=None)``
serves the audit's ``worklist.jsonl`` (see :mod:`test_catch_loss_audit_draw`) as the UI's
whole-word question, ``correct`` / ``mistake`` / ``unclear``, and the clips come from the box's
look clip cache, where #212 keeps each clip's source bytes as ``<audio_sha256>.audio``. Pinned:

* every clip is checked against its worklist row's ``audio_sha256`` before the session loads;
* the page gets the UI's blind fields only: the whole ayah text (the words joined by a space,
  from ``uthmani_words(surah_ayah)``) with the target word highlighted, never a stratum, a model's
  output, a weight, a tally or where a clip came from;
* one seeded queue, independent of the worklist file's order, with the two strata interleaved;
* the whole clip plays, as 16 kHz WAV, whatever the clip's own rate or container, and whatever
  span the row records;
* an answer is stored under the worklist row's ``id``, which is what the scorer joins on.
"""

from __future__ import annotations

import hashlib
import io
import json
import wave

import numpy as np
import pytest
import soundfile as sf

from tadabur.listening_session import read_verdicts
from tadabur.tashkeel_audit_ui import SessionState, page_key

WORDS = ("بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ", "ٱلْحَمْدُ", "لِلَّهِ", "رَبِّ",
         "ٱلْعَٰلَمِينَ", "مَٰلِكِ", "يَوْمِ")
REMOVED, ADDED = 100, 50
BLIND_FIELDS = {"key", "mode", "surah_ayah", "before", "carrier", "after", "choices", "heard",
                "note", "review"}
#: The source rates and containers the cache holds; every clip plays back at 16 kHz.
FORMATS = ((16000, "WAV"), (8000, "FLAC"), (48000, "WAV"), (16000, "FLAC"))


def _words(surah_ayah: str) -> list[str]:
    return list(WORDS)


def _frames(n: int) -> int:
    """Clip ``n``'s length in 16 kHz samples: every clip is a different length."""
    return 1600 + 16 * n


def _clip_bytes(n: int) -> bytes:
    rate, container = FORMATS[n % len(FORMATS)]
    buffer = io.BytesIO()
    samples = np.linspace(-0.5, 0.5, _frames(n) * rate // 16000, dtype=np.float32)
    sf.write(buffer, samples, rate, format=container, subtype="PCM_16")
    return buffer.getvalue()


def _row(n: int) -> dict:
    data = _clip_bytes(n)
    word_index = n % len(WORDS)
    return {"id": hashlib.sha256(f"item{n}".encode()).hexdigest()[:16], "split": "clean",
            "shard_file": f"clean-{n // 10:05d}", "row": n % 10,
            "audio_filename": f"{1 + n % 5}_1_{n:09d}.flac",
            "audio_sha256": hashlib.sha256(data).hexdigest(),
            "start_sample": 800, "end_sample": 1200,  # a span: the page plays the whole clip
            "surah_ayah": f"{1 + n % 5}:1", "word_index": word_index, "word": WORDS[word_index]}


class Audit:
    """A worklist of :data:`REMOVED` + :data:`ADDED` rows (the first :data:`REMOVED` are the
    removed stratum, which only this test knows) and a clip cache holding each one's clip."""

    def __init__(self, root):
        self.root = root
        self.cache = root / "clips"
        self.cache.mkdir(parents=True)
        self.rows = [_row(n) for n in range(REMOVED + ADDED)]
        for n, row in enumerate(self.rows):
            (self.cache / f"{row['audio_sha256']}.audio").write_bytes(_clip_bytes(n))
        self.worklist = root / "worklist.jsonl"
        self.write(self.rows)
        self.verdicts = root / "verdicts.jsonl"

    def write(self, rows) -> None:
        self.worklist.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                         for row in rows), encoding="utf-8")

    def load(self) -> SessionState:
        return SessionState.load_catch_loss(self.worklist, self.verdicts, self.cache, _words)

    def stratum(self, view: dict) -> str:
        n = [page_key(row["id"]) for row in self.rows].index(view["key"])
        return "removed" if n < REMOVED else "added"


# --- what the page is told -----------------------------------------------------------------------


def test_the_page_is_sent_the_blind_fields_and_the_ayah_with_the_word_highlighted(tmp_path):
    audit = Audit(tmp_path)
    state = audit.load()

    payload = state.payload()

    assert set(payload) == {"sites", "progress"}
    assert payload["progress"] == {"answered": 0, "total": REMOVED + ADDED}
    assert len(payload["sites"]) == REMOVED + ADDED
    by_key = {page_key(row["id"]): row for row in audit.rows}
    for view in payload["sites"]:
        row = by_key[view["key"]]
        assert set(view) == BLIND_FIELDS
        assert view["mode"] == "word"
        assert view["choices"] == ["correct", "mistake", "unclear"]
        assert view["surah_ayah"] == row["surah_ayah"]
        assert view["before"] + view["carrier"] + view["after"] == " ".join(WORDS)
        assert view["carrier"].strip() == row["word"]


def test_the_page_never_learns_the_stratum_a_model_s_output_or_where_a_clip_came_from(tmp_path):
    audit = Audit(tmp_path)

    views = audit.load().payload()["sites"]
    page = json.dumps([{name: value for name, value in view.items() if name != "key"}
                       for view in views], ensure_ascii=False)

    for word in ("removed", "added", "stratum", "probability", "weight", "c95", "candidate",
                 "flag", "cause", "grade", "model", "tally"):
        assert word not in page
    for row in audit.rows:
        for secret in (row["id"], row["audio_sha256"], row["shard_file"], row["audio_filename"]):
            assert secret not in page


def test_the_two_strata_are_interleaved_in_one_seeded_queue(tmp_path):
    audit = Audit(tmp_path)
    views = audit.load().payload()["sites"]

    strata = [audit.stratum(view) for view in views]

    assert strata[:30].count("added") > 0 and strata[:30].count("removed") > 0
    assert strata[-30:].count("added") > 0 and strata[-30:].count("removed") > 0
    assert [view["key"] for view in views] != [page_key(row["id"]) for row in audit.rows]


def test_the_queue_does_not_depend_on_the_worklist_files_order(tmp_path):
    audit = Audit(tmp_path)
    first = [view["key"] for view in audit.load().payload()["sites"]]

    audit.write(list(reversed(audit.rows)))

    assert [view["key"] for view in audit.load().payload()["sites"]] == first


# --- the clips are verified before the session loads ---------------------------------------------


def test_a_clip_whose_checksum_does_not_match_refuses_at_startup(tmp_path):
    audit = Audit(tmp_path)
    (audit.cache / f"{audit.rows[7]['audio_sha256']}.audio").write_bytes(_clip_bytes(8))

    with pytest.raises(ValueError, match="sha256|checksum"):
        audit.load()


def test_a_clip_missing_from_the_cache_refuses_at_startup(tmp_path):
    audit = Audit(tmp_path)
    (audit.cache / f"{audit.rows[7]['audio_sha256']}.audio").unlink()

    with pytest.raises((ValueError, OSError)):
        audit.load()


# --- the audio -----------------------------------------------------------------------------------


def _wav(data: bytes) -> tuple[int, int]:
    with wave.open(io.BytesIO(data)) as handle:
        return handle.getframerate(), handle.getnframes()


def test_the_whole_clip_plays_at_16_khz_whatever_its_container_rate_or_recorded_span(tmp_path):
    audit = Audit(tmp_path)
    state = audit.load()

    for n, row in enumerate(audit.rows):
        key = page_key(row["id"])
        rate, frames = _wav(state.audio(key, whole=False))
        assert rate == 16000
        assert frames == pytest.approx(_frames(n), abs=2)  # the clip, not the 400-sample span
        assert state.audio(key, whole=True) == state.audio(key, whole=False)


# --- the answers ---------------------------------------------------------------------------------


def test_an_answer_is_stored_under_the_worklist_id_and_counts_in_the_progress(tmp_path):
    audit = Audit(tmp_path)
    state = audit.load()
    row = audit.rows[3]

    state.record(page_key(row["id"]), "mistake", "", None)

    assert read_verdicts(audit.verdicts)[row["id"]].heard == "mistake"
    assert state.progress() == {"answered": 1, "total": REMOVED + ADDED}
    assert audit.load().progress() == {"answered": 1, "total": REMOVED + ADDED}  # resumed


def test_only_the_word_questions_answers_are_accepted(tmp_path):
    audit = Audit(tmp_path)
    state = audit.load()
    row = audit.rows[3]
    key = page_key(row["id"])

    state.record(key, "correct", "", None)
    state.record(key, "unclear", "", "correct")
    with pytest.raises(ValueError):
        state.record(key, "yes", "", "unclear")  # the page says "Yes"; the stored answer is "correct"
    assert read_verdicts(audit.verdicts)[row["id"]].heard == "unclear"

"""The catch-loss audit's snips through the #61 listening UI (#264).

``SessionState.load_catch_loss(worklist, verdicts, clip_cache, uthmani_words, review=None,
snips=None)`` takes the sidecar :mod:`tadabur.word_snips` builds (``test_word_snips``). Pinned:

* without ``snips`` nothing changes: both audio routes play the whole clip, byte for byte what
  the UI played before, and a verdict file is written exactly as before (no ``whole_played``);
* with it, an item whose snip is ``alignment`` plays ``samples[start:end]`` of the decoded 16 kHz
  clip, a ``whole_clip`` item plays the whole clip, and ``whole=1`` plays the whole clip for
  either, over the existing route;
* the page is told nothing more: the payload is the one it gets without snips;
* a verdict records ``whole_played``: true when the whole clip was served for the site (asked for
  with ``whole=1``, or the item's snip is ``whole_clip``) and stays true across re-answers, false
  when only a snip played; a file loaded without snips records none (``None``, left out of the
  JSON, like ``natural``);
* a snip file that does not cover the worklist, or was built for other clips, refuses at startup;
* ``--snips`` is a ``--catch-loss`` option and is documented in the UI docstring and the
  ``launch-audit-ui`` skill.
"""

from __future__ import annotations

import hashlib
import io
import json
import sys
import threading
import wave
from pathlib import Path
from urllib.request import urlopen

import numpy as np
import pytest
import soundfile as sf

from tadabur import tashkeel_audit_ui
from tadabur.audio import decode_to_mono_16k_with_rate_and_decoder
from tadabur.listening_session import Verdict, read_verdicts, write_verdicts
from tadabur.tashkeel_audit_ui import (
    SessionHandler,
    SessionState,
    build_parser,
    encode_wav,
    page_key,
)
from tadabur.audit_http import serve

WORDS = ("بِسْمِ", "ٱللَّهِ", "ٱلرَّحْمَٰنِ", "ٱلرَّحِيمِ", "ٱلْحَمْدُ", "لِلَّهِ", "رَبِّ", "مَٰلِكِ")
CLIP = 12 * 16000
#: Per row: its snip, as the sidecar holds it. Row 2 plays whole by default.
SNIPS = {0: (16000, 64000, "alignment"), 1: (32000, 96000, "alignment"),
         2: (None, None, "whole_clip"), 3: (0, 48000, "alignment")}
REPO = Path(__file__).resolve().parents[2]
#: What the page is sent for an item, with or without snips.
BLIND_FIELDS = {"key", "mode", "surah_ayah", "before", "carrier", "after", "choices", "heard",
                "note", "review"}


def _words(surah_ayah: str) -> list[str]:
    return list(WORDS)


def _clip_bytes(n: int) -> bytes:
    """A 12 s 16 kHz WAV whose every sample differs by clip, so a wrong slice cannot match."""
    samples = 0.5 * np.sin(np.arange(CLIP) * (0.01 + 0.003 * n)).astype(np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, samples, 16000, format="WAV", subtype="PCM_16")
    return buffer.getvalue()


def _row(n: int) -> dict:
    return {"id": hashlib.sha256(f"item{n}".encode()).hexdigest()[:16], "split": "clean",
            "shard_file": "clean-00000", "row": n, "audio_filename": f"1_{n}_{n:09d}.flac",
            "audio_sha256": hashlib.sha256(_clip_bytes(n)).hexdigest(),
            "start_sample": 0, "end_sample": CLIP, "surah_ayah": "1:1",
            "word_index": n + 1, "word": WORDS[n + 1]}


def _pcm(wav_bytes: bytes) -> tuple[int, bytes]:
    with wave.open(io.BytesIO(wav_bytes)) as handle:
        return handle.getnframes(), handle.readframes(handle.getnframes())


class Audit:
    """Four worklist rows, their clips in a cache, and a snip file covering them."""

    def __init__(self, root: Path):
        self.root = root
        self.cache = root / "clips"
        self.cache.mkdir(parents=True)
        self.rows = [_row(n) for n in range(4)]
        for n, row in enumerate(self.rows):
            (self.cache / f"{row['audio_sha256']}.audio").write_bytes(_clip_bytes(n))
        self.worklist = root / "worklist.jsonl"
        self.worklist.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n"
                                         for row in self.rows), encoding="utf-8")
        self.verdicts = root / "verdicts.jsonl"
        self.snips = root / "snips.jsonl"
        self.write_snips()

    def write_snips(self, rows=None) -> None:
        rows = rows if rows is not None else [
            {"id": row["id"], "audio_sha256": row["audio_sha256"], "start_sample": span[0],
             "end_sample": span[1], "snip_source": span[2]}
            for row, span in zip(self.rows, SNIPS.values())]
        self.snips.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")

    def load(self, *, snips: bool = True) -> SessionState:
        args = (self.worklist, self.verdicts, self.cache, _words)
        if snips:
            return SessionState.load_catch_loss(*args, snips=self.snips)
        return SessionState.load_catch_loss(*args)  # the call every existing caller makes

    def key(self, n: int) -> str:
        return page_key(self.rows[n]["id"])

    def verdict_line(self, n: int) -> dict:
        lines = self.verdicts.read_text(encoding="utf-8").splitlines()
        return next(json.loads(line) for line in lines
                    if json.loads(line)["site_id"] == self.rows[n]["id"])


# --- without --snips nothing changes --------------------------------------------------------


def test_without_snips_both_routes_play_the_whole_clip_byte_for_byte_as_before(tmp_path):
    audit = Audit(tmp_path)
    state = audit.load(snips=False)

    for n in range(4):
        before = encode_wav(decode_to_mono_16k_with_rate_and_decoder(_clip_bytes(n))[0])
        assert state.audio(audit.key(n), whole=False) == before
        assert state.audio(audit.key(n), whole=True) == before


def test_without_snips_a_verdict_is_written_exactly_as_before(tmp_path):
    audit = Audit(tmp_path)
    state = audit.load(snips=False)

    state.record(audit.key(0), "mistake", "", None)
    state.audio(audit.key(0), whole=True)  # asking for the whole clip changes nothing here

    assert audit.verdicts.read_text(encoding="utf-8") == (
        json.dumps({"heard": "mistake", "note": "", "site_id": audit.rows[0]["id"]}) + "\n")
    assert read_verdicts(audit.verdicts)[audit.rows[0]["id"]].whole_played is None


# --- with them, the item plays its snip -----------------------------------------------------


def test_an_alignment_snip_plays_exactly_its_slice_of_the_decoded_clip(tmp_path):
    audit = Audit(tmp_path)
    state = audit.load()
    for n in (0, 1, 3):
        start, end, _ = SNIPS[n]
        frames, whole = _pcm(state.audio(audit.key(n), whole=True))

        snip_frames, snip = _pcm(state.audio(audit.key(n), whole=False))

        assert frames == CLIP
        assert snip_frames == end - start
        assert snip == whole[2 * start:2 * end]


def test_a_whole_clip_snip_plays_the_whole_clip_by_default(tmp_path):
    audit = Audit(tmp_path)
    state = audit.load()

    assert state.audio(audit.key(2), whole=False) == state.audio(audit.key(2), whole=True)
    assert _pcm(state.audio(audit.key(2), whole=False))[0] == CLIP


def test_the_whole_clip_control_plays_the_same_bytes_with_or_without_snips(tmp_path):
    audit = Audit(tmp_path)
    with_snips, without = audit.load(), audit.load(snips=False)

    for n in range(4):
        assert with_snips.audio(audit.key(n), whole=True) == without.audio(audit.key(n), whole=True)


def test_the_whole_clip_control_works_over_the_existing_route(tmp_path):
    audit = Audit(tmp_path)
    state = audit.load()
    server = serve(SessionHandler, state, 0)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}/api/audio?key={audit.key(0)}"
        with urlopen(f"{base}&whole=0", timeout=10) as snip:
            snip_body = snip.read()
        with urlopen(f"{base}&whole=1", timeout=10) as whole:
            whole_body, kind = whole.read(), whole.headers["Content-Type"]
    finally:
        server.shutdown()
        server.server_close()

    assert kind == "audio/wav"
    assert _pcm(snip_body)[0] == SNIPS[0][1] - SNIPS[0][0]
    assert _pcm(whole_body)[0] == CLIP
    state.record(audit.key(0), "correct", "", None)
    assert audit.verdict_line(0)["whole_played"] is True


# --- the page is blind ----------------------------------------------------------------------


def test_the_page_is_sent_what_it_is_sent_without_snips(tmp_path):
    audit = Audit(tmp_path)

    payload = audit.load().payload()

    assert payload == audit.load(snips=False).payload()
    assert all(set(view) == BLIND_FIELDS for view in payload["sites"])
    text = json.dumps(payload, ensure_ascii=False)
    for secret in ("snip", "alignment", "whole_clip", "whole_played", "start_sample", "model"):
        assert secret not in text


# --- the verdict records whether the whole clip played --------------------------------------


def test_a_verdict_after_only_the_snip_played_records_whole_played_false(tmp_path):
    audit = Audit(tmp_path)
    state = audit.load()
    state.audio(audit.key(0), whole=False)

    state.record(audit.key(0), "correct", "", None)

    assert audit.verdict_line(0)["whole_played"] is False
    assert read_verdicts(audit.verdicts)[audit.rows[0]["id"]].whole_played is False


def test_playing_the_whole_clip_before_answering_records_it_for_that_site_alone(tmp_path):
    audit = Audit(tmp_path)
    state = audit.load()
    state.audio(audit.key(0), whole=True)

    state.record(audit.key(0), "mistake", "", None)
    state.record(audit.key(1), "mistake", "", None)

    assert audit.verdict_line(0)["whole_played"] is True
    assert audit.verdict_line(1)["whole_played"] is False


def test_an_item_whose_snip_is_the_whole_clip_records_that_the_whole_clip_played(tmp_path):
    audit = Audit(tmp_path)
    state = audit.load()

    state.record(audit.key(2), "correct", "", None)

    assert audit.verdict_line(2)["whole_played"] is True


def test_whole_played_stays_true_across_a_re_answer_and_a_restart(tmp_path):
    audit = Audit(tmp_path)
    state = audit.load()
    state.audio(audit.key(0), whole=True)
    state.record(audit.key(0), "mistake", "", None)

    state.record(audit.key(0), "correct", "", "mistake")
    assert audit.verdict_line(0)["whole_played"] is True

    restarted = audit.load()  # nothing served yet in this process
    restarted.audio(audit.key(0), whole=False)
    restarted.record(audit.key(0), "unclear", "", "correct")
    assert audit.verdict_line(0)["whole_played"] is True
    assert read_verdicts(audit.verdicts)[audit.rows[0]["id"]].heard == "unclear"


def test_a_verdict_carries_whole_played_only_when_it_is_known():
    assert "whole_played" not in Verdict("a", "correct").as_json()
    assert Verdict("a", "correct", whole_played=False).as_json()["whole_played"] is False
    assert Verdict("a", "correct", whole_played=True).as_json()["whole_played"] is True


def test_the_verdicts_file_round_trips_whole_played_and_still_reads_the_old_shape(tmp_path):
    path = tmp_path / "verdicts.jsonl"
    verdicts = {"a": Verdict("a", "correct"), "b": Verdict("b", "mistake", "n", whole_played=True),
                "c": Verdict("c", "unclear", whole_played=False)}

    write_verdicts(verdicts, path)

    assert read_verdicts(path) == verdicts
    assert "whole_played" not in path.read_text(encoding="utf-8").splitlines()[0]


def test_a_whole_played_that_is_not_a_boolean_is_refused(tmp_path):
    path = tmp_path / "verdicts.jsonl"
    path.write_text(json.dumps({"site_id": "a", "heard": "correct", "note": "",
                                "whole_played": "yes"}) + "\n", encoding="utf-8")

    with pytest.raises(ValueError):
        read_verdicts(path)


# --- the snip file is checked before the session loads --------------------------------------


def test_a_snip_file_that_misses_a_worklist_id_refuses_at_startup(tmp_path):
    audit = Audit(tmp_path)
    audit.write_snips([
        {"id": row["id"], "audio_sha256": row["audio_sha256"], "start_sample": None,
         "end_sample": None, "snip_source": "whole_clip"} for row in audit.rows[:3]])

    with pytest.raises(ValueError, match=audit.rows[3]["id"]):
        audit.load()


def test_a_snip_built_for_another_clip_refuses_at_startup(tmp_path):
    audit = Audit(tmp_path)
    rows = [{"id": row["id"], "audio_sha256": row["audio_sha256"], "start_sample": None,
             "end_sample": None, "snip_source": "whole_clip"} for row in audit.rows]
    rows[1]["audio_sha256"] = "0" * 64
    audit.write_snips(rows)

    with pytest.raises(ValueError, match=audit.rows[1]["id"]):
        audit.load()


# --- the option and its documentation -------------------------------------------------------


def test_snips_is_an_option_and_needs_catch_loss(monkeypatch, tmp_path):
    assert build_parser().parse_args(["--snips", "s.jsonl"]).snips == Path("s.jsonl")

    monkeypatch.setattr(sys, "argv", ["tashkeel_audit_ui", "--audio-dir", str(tmp_path),
                                      "--snips", "s.jsonl"])
    with pytest.raises(SystemExit):
        tashkeel_audit_ui.main()


def test_the_ui_docstring_and_the_launch_skill_document_snips():
    skill = (REPO / ".claude" / "skills" / "launch-audit-ui" / "SKILL.md").read_text("utf-8")

    for text in (tashkeel_audit_ui.__doc__, skill):
        assert "--snips" in text
        assert "whole" in text.lower()

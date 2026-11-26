"""The S5 listening UI's session: blind page, committed order, verdicts file, audio (#311 PR3).

Committed before ``tadabur/followalong_audit/s5_ui.py`` exists and **locked**: later turns may add
tests but never weaken or rewrite these. They pin the brief's PR3 criteria and the PR2-locked
verdicts schema (``test_followalong_score_verdicts.py``): the UI serves ``pilot.jsonl`` then (behind
an explicit flag) ``main.jsonl`` in their committed order from the blind worklist, plays snips with
a whole-clip control, never tells the page a stratum, a grade, an arm or a weight, and writes
``verdicts.jsonl`` exactly as ``s5_score.read_verdicts`` reads it.

Contract this file pins (``tadabur.followalong_audit.s5_ui``)
-----------------------------------------------------------
* ``read_audio_manifest(path) -> {audio_sha256: row}``: a gzipped JSONL whose rows carry
  ``audio_sha256``, ``wav_sha256`` and ``samples`` (the committed ``app_arm/audio_manifest.jsonl.gz``).
* ``load_rows(worklist, snips, orders, *, uthmani_words, references=None, sittings=("pilot",))
  -> list[CatchLossRow]``: the worklist's rows in served order. ``orders`` maps ``"pilot"`` /
  ``"main"`` to ``{"id", "position"}`` JSONL files; the served order is each served sitting's file
  in turn, in file order (a repeat is its own id). Refused (``ValueError``): a row whose word is
  not the ayah's (``catch_loss_worklist.read_catch_loss_worklist``), an order id the worklist lacks,
  a worklist id served twice, a served id with no snip or a snip built for another clip or running
  past its clip (``tashkeel_audit_ui.verify_snips``), and, with ``references`` (``label ->`` an object
  with ``word_of``), a row whose engine word is recited together with another canonical word
  (a wasl-merged word): ``word_of[word_index]`` shared with any other index.
* ``SessionState.load(*, worklist, snips, orders, wav_dir, wav_manifest, verdicts, uthmani_words,
  references=None, sittings=("pilot",))``: ``load_rows`` plus the audio: every served clip's
  ``<wav_dir>/<audio_sha256>.wav`` is verified against the manifest (``clips.verify_wav``) and an
  existing verdicts file is read with ``read_verdicts`` (refused when invalid).
* ``state.payload() -> {"sites", "progress", "vocabulary"}``: ``progress`` is ``{"answered",
  "total"}`` over the served rows; ``vocabulary`` is ``{"verdicts", "kinds", "sub_reasons"}`` (the
  ``s5_score`` tuples, as lists); each site is exactly ``{"key", "surah_ayah", "before", "carrier",
  "after", "answer"}``: ``key`` is ``tashkeel_audit_ui.page_key(id)``, ``before + carrier + after``
  the ayah with the word highlighted, ``answer`` ``None`` or ``{"verdict", "kinds", "sub_reason",
  "note"}``.
* ``state.record(key, verdict, kinds, sub_reason, note, previous) -> dict``: saves the row (the
  schema of ``s5_score``; ``whole_played`` is true when the snip is the whole clip, when the whole
  clip has been served for the item in any session, or already was) replacing the answer the page
  last saw (``previous``, an ``answer`` value): ``StaleAnswer`` when it differs from the stored one,
  ``KeyError`` for a key not served, ``ValueError`` for an answer ``read_verdicts`` would refuse. The
  file is rewritten whole, sorted by id, atomically; rows it holds for ids not served are kept.
* ``state.audio(key, whole) -> bytes``: a 16 kHz mono 16-bit WAV: the item's snip samples, or the
  whole clip when ``whole`` is true or the snip is the whole clip. Serving the whole clip is
  remembered for the verdict and rewrites one already given.
* ``Handler`` (an ``audit_http.AuditHandler``): ``GET /`` the page, ``GET /api/sites``,
  ``GET /api/audio?key=&whole=0|1``, ``POST /api/verdict`` (``{key, verdict, kinds, sub_reason,
  note, previous}`` -> ``{"progress", "answer"}``; 400 invalid, 404 unknown key, 409 stale, 500 not
  saved); any other path is 404.
* ``build_parser()``: ``--worklist --snips --pilot --main`` (defaults: the committed ``s5_draw/``
  files), ``--wav-dir`` and ``--verdicts`` (required), ``--wav-manifest`` (default the committed
  audio manifest), ``--with-main`` (serve the main sitting too), ``--port`` (default 8000), ``--host``
  (default ``127.0.0.1``).
"""

from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import threading
import wave
from pathlib import Path

import numpy as np
import pytest

from tadabur.audit_http import serve
from tadabur.catch_loss_worklist import WORKLIST_FIELDS
from tadabur.followalong_audit.s5_score import (
    MISTAKE_KINDS,
    SUB_REASONS,
    VERDICTS,
    read_verdicts,
)
from tadabur.followalong_audit.s5_ui import (
    Handler,
    SessionState,
    StaleAnswer,
    build_parser,
    load_rows,
    read_audio_manifest,
)
from tadabur.tashkeel_audit_ui import page_key
from tadabur.word_snips import ALIGNMENT, WHOLE_CLIP, Snip, write_snips

WORDS = ["ٱلْحَمْدُ", "لِلَّهِ", "رَبِّ", "ٱلْعَـٰلَمِينَ", "ٱلرَّحْمَـٰنِ"]
LABEL = "1:2"
CLIP_SAMPLES = 40_000
SNIP = (10_000, 22_000)


def uthmani(label: str) -> list[str]:
    assert label == LABEL
    return list(WORDS)


class Draw:
    """A synthetic draw: five pilot originals plus one repeat, three main items, their WAVs."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.wavs = root / "wav"
        self.wavs.mkdir()
        self.rows: list[dict] = []
        self.snips: list[Snip] = []
        for n in range(8):
            digest = hashlib.sha256(f"clip{n}".encode()).hexdigest()
            self._wav(digest, n)
            self.rows.append(self._row(n, digest, f"{n:016x}"))
        # a repeat re-asks item 0 under its own id; item 4 is an S5-edge word that plays whole
        self.rows.append(self._row(0, self.rows[0]["audio_sha256"], "00000000000000ff"))
        for row in self.rows:
            whole = row["id"] == f"{4:016x}"
            self.snips.append(Snip(row["id"], row["audio_sha256"], *(
                (None, None) if whole else SNIP), WHOLE_CLIP if whole else ALIGNMENT))
        self.pilot_ids = [*(f"{n:016x}" for n in (3, 0, 4)), "00000000000000ff",
                          *(f"{n:016x}" for n in (1, 2))]
        self.main_ids = [f"{n:016x}" for n in (7, 5, 6)]
        self.worklist = root / "worklist.jsonl"
        self.worklist.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n"
                                         for r in sorted(self.rows, key=lambda r: r["id"])),
                                 encoding="utf-8")
        self.snip_file = root / "snips.jsonl"
        write_snips(self.snips, self.snip_file)
        self.orders = {"pilot": self._order("pilot", self.pilot_ids),
                       "main": self._order("main", self.main_ids)}
        self.manifest = root / "audio_manifest.jsonl.gz"
        self._write_manifest()
        self.verdicts = root / "verdicts.jsonl"

    def samples(self, n: int) -> list[int]:
        return ((np.arange(CLIP_SAMPLES) * 7 + n) % 20_000).tolist()

    def _wav(self, digest: str, n: int) -> None:
        with wave.open(str(self.wavs / f"{digest}.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16_000)
            handle.writeframes(np.asarray(self.samples(n), dtype="<i2").tobytes())

    def _row(self, n: int, digest: str, ident: str) -> dict:
        return {"id": ident, "split": "clean", "shard_file": f"clean-0000{n}-of-00411", "row": n,
                "audio_filename": f"1_2_{100000 + n}.flac", "audio_sha256": digest,
                "start_sample": 0, "end_sample": CLIP_SAMPLES, "surah_ayah": LABEL,
                "word_index": n % len(WORDS), "word": WORDS[n % len(WORDS)]}

    def _order(self, name: str, ids: list[str]) -> Path:
        path = self.root / f"{name}.jsonl"
        path.write_text("".join(json.dumps({"id": ident, "position": place}) + "\n"
                                for place, ident in enumerate(ids, 1)), encoding="utf-8")
        return path

    def _write_manifest(self) -> None:
        lines = []
        for digest in sorted({row["audio_sha256"] for row in self.rows}):
            wav = (self.wavs / f"{digest}.wav").read_bytes()
            lines.append({"audio_sha256": digest, "wav_sha256": hashlib.sha256(wav).hexdigest(),
                          "samples": CLIP_SAMPLES})
        with gzip.open(self.manifest, "wt", encoding="utf-8") as sink:
            sink.write("".join(json.dumps(line) + "\n" for line in lines))

    def row(self, ident: str) -> dict:
        return next(row for row in self.rows if row["id"] == ident)

    def kwargs(self, **changes) -> dict:
        base = {"worklist": self.worklist, "snips": self.snip_file, "orders": self.orders,
                "wav_dir": self.wavs, "wav_manifest": self.manifest, "verdicts": self.verdicts,
                "uthmani_words": uthmani}
        return {**base, **changes}

    def load(self, **changes) -> SessionState:
        return SessionState.load(**self.kwargs(**changes))


@pytest.fixture
def draw(tmp_path) -> Draw:
    return Draw(tmp_path)


def key_of(ident: str) -> str:
    return page_key(ident)


def samples_of(wav: bytes) -> list[int]:
    import io

    with wave.open(io.BytesIO(wav), "rb") as handle:
        assert (handle.getnchannels(), handle.getsampwidth(), handle.getframerate()) == (
            1, 2, 16_000)
        raw = handle.readframes(handle.getnframes())
    return np.frombuffer(raw, dtype="<i2").tolist()


def mistake(state, ident, kinds=("letter",), previous=None, note=""):
    return state.record(key_of(ident), "mistake", list(kinds), None, note, previous)


def answered(state, ident):
    return next(site for site in state.payload()["sites"] if site["key"] == key_of(ident))["answer"]


# --- the rows -------------------------------------------------------------------------------------
def test_the_manifest_is_read_by_audio_sha256(draw):
    manifest = read_audio_manifest(draw.manifest)
    assert set(manifest) == {row["audio_sha256"] for row in draw.rows}
    assert all({"wav_sha256", "samples"} <= set(entry) for entry in manifest.values())


def test_the_pilot_alone_is_served_in_its_committed_order_repeats_included(draw):
    rows = load_rows(draw.worklist, draw.snip_file, draw.orders, uthmani_words=uthmani)
    assert [row.site.site_id for row in rows] == draw.pilot_ids
    assert len(rows) == 6


def test_the_main_sitting_follows_the_pilot_only_behind_the_explicit_sittings(draw):
    rows = load_rows(draw.worklist, draw.snip_file, draw.orders, uthmani_words=uthmani,
                     sittings=("pilot", "main"))
    assert [row.site.site_id for row in rows] == draw.pilot_ids + draw.main_ids


def test_an_unserved_sitting_is_neither_listed_nor_answerable(draw):
    state = draw.load()
    assert state.payload()["progress"] == {"answered": 0, "total": 6}
    with pytest.raises(KeyError):
        mistake(state, draw.main_ids[0])
    both = draw.load(sittings=("pilot", "main"))
    assert both.payload()["progress"]["total"] == 9
    mistake(both, draw.main_ids[0])


@pytest.mark.parametrize("breakage", ["order_unknown", "no_snip", "other_clip", "past_clip",
                                      "word", "double"])
def test_a_draw_that_does_not_hold_together_is_refused_at_startup(draw, breakage):
    kwargs = draw.kwargs()
    if breakage == "order_unknown":
        draw.orders["pilot"].write_text(
            json.dumps({"id": "ffffffffffffffff", "position": 1}) + "\n", encoding="utf-8")
    elif breakage == "no_snip":
        write_snips([snip for snip in draw.snips if snip.site_id != draw.pilot_ids[0]],
                    draw.snip_file)
    elif breakage == "other_clip":
        write_snips([Snip(snip.site_id, "0" * 64, snip.start_sample, snip.end_sample,
                          snip.snip_source) if snip.site_id == draw.pilot_ids[0] else snip
                     for snip in draw.snips], draw.snip_file)
    elif breakage == "past_clip":
        write_snips([Snip(snip.site_id, snip.audio_sha256, 1000, CLIP_SAMPLES + 1, ALIGNMENT)
                     if snip.site_id == draw.pilot_ids[0] else snip for snip in draw.snips],
                    draw.snip_file)
    elif breakage == "word":
        kwargs["uthmani_words"] = lambda label: list(reversed(WORDS))
    else:
        draw.orders["pilot"].write_text(
            "".join(json.dumps({"id": draw.pilot_ids[0], "position": p}) + "\n" for p in (1, 2)),
            encoding="utf-8")
    with pytest.raises(ValueError):
        SessionState.load(**kwargs)


class _Reference:
    def __init__(self, word_of):
        self.word_of = tuple(word_of)


def test_a_wasl_merged_drawn_word_fails_startup_loudly(draw):
    plain = lambda label: _Reference(range(1, 6))  # noqa: E731
    assert draw.load(references=plain).payload()["progress"]["total"] == 6
    merged = lambda label: _Reference((1, 2, 2, 3, 4))  # noqa: E731
    with pytest.raises(ValueError):
        draw.load(references=merged)


# --- what the page is told ------------------------------------------------------------------------
def test_the_page_gets_exactly_the_blind_fields_in_served_order(draw):
    payload = draw.load().payload()
    assert set(payload) == {"sites", "progress", "vocabulary"}
    assert payload["vocabulary"] == {"verdicts": list(VERDICTS), "kinds": list(MISTAKE_KINDS),
                                     "sub_reasons": list(SUB_REASONS)}
    assert [site["key"] for site in payload["sites"]] == [key_of(i) for i in draw.pilot_ids]
    for site in payload["sites"]:
        row = draw.row(next(i for i in draw.pilot_ids if key_of(i) == site["key"]))
        assert set(site) == {"key", "surah_ayah", "before", "carrier", "after", "answer"}
        assert site["surah_ayah"] == LABEL and site["answer"] is None
        assert site["before"] + site["carrier"] + site["after"] == " ".join(WORDS)
        assert site["carrier"].strip() == row["word"]


def test_the_page_never_learns_a_stratum_a_grade_an_arm_a_weight_or_where_a_clip_came_from(draw):
    state = draw.load(sittings=("pilot", "main"))
    page = json.dumps([{name: value for name, value in site.items() if name != "key"}
                       for site in state.payload()["sites"]], ensure_ascii=False)
    for word in ("stratum", "probability", "weight", "cell", "pilot", "main", "repeat", "whole",
                 "snip", "alignment", "teacher", "student", "oracle", "model", "flag", "cause",
                 "grade", "S1", "S3", "S4", "S5"):
        assert word not in page
    for row in draw.rows:
        for secret in (row["id"], row["audio_sha256"], row["shard_file"], row["audio_filename"]):
            assert secret not in page
    assert not set(WORKLIST_FIELDS) & {"stratum", "pilot"}


def test_a_repeat_is_its_own_item_and_the_original_does_not_answer_it(draw):
    state = draw.load()
    assert key_of("00000000000000ff") != key_of(f"{0:016x}")
    mistake(state, f"{0:016x}")
    assert answered(state, "00000000000000ff") is None
    assert answered(state, f"{0:016x}")["verdict"] == "mistake"


# --- the verdicts file ----------------------------------------------------------------------------
def test_an_answer_is_stored_exactly_as_the_scorer_reads_it(draw):
    state = draw.load()
    state.record(key_of(draw.pilot_ids[0]), "correct", [], None, "fine", None)
    mistake(state, draw.pilot_ids[1], ("letter", "added"), note="س")
    state.record(key_of(draw.pilot_ids[2]), "unusable audio", [], "not_in_clip", "", None)
    state.record(key_of(draw.pilot_ids[3]), "not sure", [], None, "", None)
    state.record(key_of(draw.pilot_ids[4]), "different text", [], None, "", None)
    stored = read_verdicts(draw.verdicts, draw.rows)
    assert set(stored) == set(draw.pilot_ids[:5])
    assert stored[draw.pilot_ids[1]] == {
        "id": draw.pilot_ids[1], "verdict": "mistake", "kinds": ["letter", "added"],
        "sub_reason": None, "whole_played": False, "note": "س"}
    assert stored[draw.pilot_ids[2]]["sub_reason"] == "not_in_clip"
    assert state.payload()["progress"] == {"answered": 5, "total": 6}


def test_the_file_is_sorted_by_id_and_keeps_rows_for_ids_not_served(draw):
    other = {"id": draw.main_ids[0], "verdict": "correct", "kinds": [], "sub_reason": None,
             "whole_played": True, "note": "earlier"}
    draw.verdicts.write_text(json.dumps(other) + "\n", encoding="utf-8")
    state = draw.load()
    for ident in draw.pilot_ids[:3]:
        state.record(key_of(ident), "correct", [], None, "", None)
    ids = [json.loads(line)["id"] for line in draw.verdicts.read_text().splitlines()]
    assert ids == sorted(ids) and draw.main_ids[0] in ids and len(ids) == 4


def test_a_session_resumes_from_the_file_and_progress_counts_only_served_answers(draw):
    state = draw.load()
    mistake(state, draw.pilot_ids[0])
    again = draw.load()
    assert answered(again, draw.pilot_ids[0])["verdict"] == "mistake"
    assert again.payload()["progress"] == {"answered": 1, "total": 6}


def test_a_verdicts_file_the_scorer_would_refuse_stops_startup(draw):
    draw.verdicts.write_text(json.dumps({"id": draw.pilot_ids[0], "verdict": "maybe"}) + "\n",
                             encoding="utf-8")
    with pytest.raises(ValueError):
        draw.load()


@pytest.mark.parametrize("verdict, kinds, sub_reason", [
    ("maybe", [], None), ("mistake", [], None), ("mistake", ["typo"], None),
    ("mistake", ["letter", "letter"], None), ("correct", ["letter"], None),
    ("correct", [], "not_in_clip"), ("unusable audio", [], "noisy")])
def test_an_answer_the_schema_forbids_is_refused_and_nothing_is_written(
        draw, verdict, kinds, sub_reason):
    state = draw.load()
    with pytest.raises(ValueError):
        state.record(key_of(draw.pilot_ids[0]), verdict, kinds, sub_reason, "", None)
    assert not draw.verdicts.exists()
    assert state.payload()["progress"]["answered"] == 0


def test_an_unknown_key_is_refused(draw):
    with pytest.raises(KeyError):
        draw.load().record("nope", "correct", [], None, "", None)


def test_a_re_answer_replaces_the_answer_the_page_saw_and_a_stale_one_is_refused(draw):
    state = draw.load()
    ident = draw.pilot_ids[0]
    mistake(state, ident)
    seen = answered(state, ident)
    state.record(key_of(ident), "correct", [], None, "", seen)
    assert answered(state, ident)["verdict"] == "correct"
    with pytest.raises(StaleAnswer):
        state.record(key_of(ident), "not sure", [], None, "", seen)
    with pytest.raises(StaleAnswer):
        state.record(key_of(draw.pilot_ids[1]), "correct", [], None, "", {"verdict": "correct"})
    assert len(read_verdicts(draw.verdicts, draw.rows)) == 1


# --- the audio ------------------------------------------------------------------------------------
def test_an_item_plays_its_snip_and_the_whole_control_plays_the_clip(draw):
    state = draw.load()
    ident = draw.pilot_ids[0]
    row = draw.row(ident)
    n = int(row["row"])
    clip = draw.samples(n)
    assert samples_of(state.audio(key_of(ident), False)) == clip[SNIP[0]:SNIP[1]]
    assert samples_of(state.audio(key_of(ident), True)) == clip


def test_a_whole_clip_snip_plays_whole_either_way_and_counts_as_whole_played(draw):
    state = draw.load()
    ident = f"{4:016x}"
    assert len(samples_of(state.audio(key_of(ident), False))) == CLIP_SAMPLES
    assert len(samples_of(state.audio(key_of(ident), True))) == CLIP_SAMPLES
    state.record(key_of(ident), "correct", [], None, "", None)
    assert read_verdicts(draw.verdicts, draw.rows)[ident]["whole_played"] is True


def test_whole_played_is_false_for_a_snip_and_true_once_the_whole_clip_has_played(draw):
    state = draw.load()
    ident = draw.pilot_ids[0]
    mistake(state, ident)
    assert read_verdicts(draw.verdicts, draw.rows)[ident]["whole_played"] is False
    state.audio(key_of(ident), True)  # checked with the Whole control after answering
    assert read_verdicts(draw.verdicts, draw.rows)[ident]["whole_played"] is True
    state.record(key_of(ident), "correct", [], None, "", answered(state, ident))
    assert read_verdicts(draw.verdicts, draw.rows)[ident]["whole_played"] is True
    assert answered(draw.load(), ident)["verdict"] == "correct"
    assert read_verdicts(draw.verdicts, draw.rows)[ident]["whole_played"] is True


def test_the_whole_clip_played_before_answering_is_recorded_with_the_answer(draw):
    state = draw.load()
    ident = draw.pilot_ids[1]
    state.audio(key_of(ident), True)
    assert not draw.verdicts.exists()
    mistake(state, ident)
    assert read_verdicts(draw.verdicts, draw.rows)[ident]["whole_played"] is True


def test_an_unknown_key_has_no_audio(draw):
    with pytest.raises(KeyError):
        draw.load().audio("nope", False)


@pytest.mark.parametrize("damage", ["missing", "bytes", "length"])
def test_a_wav_the_manifest_does_not_vouch_for_stops_startup(draw, damage):
    digest = draw.row(draw.pilot_ids[0])["audio_sha256"]
    path = draw.wavs / f"{digest}.wav"
    if damage == "missing":
        path.unlink()
    elif damage == "bytes":
        data = bytearray(path.read_bytes())
        data[-1] ^= 0xFF
        path.write_bytes(bytes(data))
    else:
        manifest = read_audio_manifest(draw.manifest)
        manifest[digest]["samples"] += 1
        with gzip.open(draw.manifest, "wt", encoding="utf-8") as sink:
            sink.write("".join(json.dumps(entry) + "\n" for entry in manifest.values()))
    with pytest.raises(ValueError):
        draw.load()


def test_a_main_clip_is_not_verified_until_the_main_sitting_is_served(draw):
    digest = draw.row(draw.main_ids[0])["audio_sha256"]
    (draw.wavs / f"{digest}.wav").unlink()
    assert draw.load().payload()["progress"]["total"] == 6
    with pytest.raises(ValueError):
        draw.load(sittings=("pilot", "main"))


# --- the HTTP routes ------------------------------------------------------------------------------
@pytest.fixture
def server(draw):
    state = draw.load()
    httpd = serve(Handler, state, 0, "127.0.0.1")
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield state, httpd.server_address[1]
    httpd.shutdown()
    httpd.server_close()


def call(port, method, path, body=None, raw=None):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    data = raw if raw is not None else (None if body is None else json.dumps(body))
    connection.request(method, path, body=data)
    response = connection.getresponse()
    payload = response.read()
    connection.close()
    return response.status, response.getheader("Content-Type"), payload


def test_the_page_is_served_and_carries_nothing_of_the_draw(server):
    _, port = server
    status, kind, page = call(port, "GET", "/")
    assert status == 200 and kind.startswith("text/html")
    text = page.decode("utf-8")
    assert 'id="whole"' in text and "<audio" in text
    for word in ("stratum", "probability", "weight", "pilot", "repeat", "oracle", "teacher",
                 "student", "key.jsonl"):
        assert word not in text


def test_the_site_list_and_the_audio_are_served(server, draw):
    state, port = server
    status, kind, body = call(port, "GET", "/api/sites")
    assert status == 200 and kind.startswith("application/json")
    assert json.loads(body) == state.payload()
    key = key_of(draw.pilot_ids[0])
    status, kind, wav = call(port, "GET", f"/api/audio?key={key}&whole=0")
    assert (status, kind) == (200, "audio/wav") and len(samples_of(wav)) == SNIP[1] - SNIP[0]
    status, _, wav = call(port, "GET", f"/api/audio?key={key}&whole=1")
    assert status == 200 and len(samples_of(wav)) == CLIP_SAMPLES
    assert call(port, "GET", "/api/audio?key=nope&whole=0")[0] == 404


def test_an_answer_posts_and_the_progress_comes_back(server, draw):
    state, port = server
    key = key_of(draw.pilot_ids[0])
    status, _, body = call(port, "POST", "/api/verdict", {
        "key": key, "verdict": "mistake", "kinds": ["haraka"], "sub_reason": None, "note": "x",
        "previous": None})
    reply = json.loads(body)
    assert status == 200 and reply["progress"] == {"answered": 1, "total": 6}
    assert reply["answer"] == {"verdict": "mistake", "kinds": ["haraka"], "sub_reason": None,
                               "note": "x"}
    assert read_verdicts(draw.verdicts, draw.rows)[draw.pilot_ids[0]]["kinds"] == ["haraka"]


def test_a_bad_stale_or_unknown_answer_gets_the_right_status(server, draw):
    state, port = server
    key = key_of(draw.pilot_ids[0])
    good = {"key": key, "verdict": "correct", "kinds": [], "sub_reason": None, "note": "",
            "previous": None}
    assert call(port, "POST", "/api/verdict", {**good, "verdict": "maybe"})[0] == 400
    assert call(port, "POST", "/api/verdict", raw="not json")[0] == 400
    assert call(port, "POST", "/api/verdict", {**good, "key": "nope"})[0] == 404
    assert call(port, "POST", "/api/verdict", good)[0] == 200
    assert call(port, "POST", "/api/verdict", {**good, "verdict": "not sure"})[0] == 409


def test_any_other_path_is_not_found(server):
    _, port = server
    assert call(port, "GET", "/api/key")[0] == 404
    assert call(port, "GET", "/key.jsonl")[0] == 404
    assert call(port, "POST", "/api/other", {})[0] == 404


# --- the command line -----------------------------------------------------------------------------
def test_the_parser_defaults_to_the_committed_draw_the_pilot_alone_and_port_8000():
    args = build_parser().parse_args(["--wav-dir", "/w", "--verdicts", "/v.jsonl"])
    assert args.port == 8000 and args.host == "127.0.0.1" and args.with_main is False
    for name in ("worklist", "snips", "pilot", "main", "wav_manifest"):
        assert getattr(args, name).exists(), name
    assert args.worklist.name == "worklist.jsonl" and args.snips.name == "snips.jsonl"
    assert args.pilot.name == "pilot.jsonl" and args.main.name == "main.jsonl"
    assert args.wav_manifest.name == "audio_manifest.jsonl.gz"


def test_the_main_sitting_needs_its_flag_and_the_wavs_and_verdicts_are_required():
    args = build_parser().parse_args(["--wav-dir", "/w", "--verdicts", "/v", "--with-main",
                                      "--port", "8002", "--host", "0.0.0.0"])
    assert (args.with_main, args.port, args.host) == (True, 8002, "0.0.0.0")
    for missing in (["--verdicts", "/v"], ["--wav-dir", "/w"]):
        with pytest.raises(SystemExit):
            build_parser().parse_args(missing)

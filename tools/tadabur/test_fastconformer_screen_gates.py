"""Tests for #267's gates (steps 1-2): each is a refusal with a message, not a judgement.

* identity: the revision and the full 64-hex ``.nemo`` SHA-256 (``d29d19d7…3e31``, the LFS oid at revision ``7f32349d``) are pinned;
  matching only the prefix and suffix the issue quotes is not a match;
* mapping gate: at least 95% of sites and edits located;
* card sanity: CTC WER on a 200-clip EveryAyah sample, stop only above 2x the card's 7.61.
"""

from __future__ import annotations

import json
import re

import pytest

from tadabur.fastconformer_screen.card_line import measure, record, sample_rows, wer
from tadabur.fastconformer_screen.gates import (
    CARD_CLIPS,
    CARD_WER,
    NEMO_SHA256,
    REVISION,
    ScreenStop,
    card_gate,
    mapping_gate,
    require_identity,
)


def test_screen_stop_is_a_runtime_error():
    assert issubclass(ScreenStop, RuntimeError)


# --- identity ---------------------------------------------------------------------------------


def test_the_pinned_identity_is_complete():
    assert re.fullmatch(r"[0-9a-f]{64}", NEMO_SHA256)
    assert NEMO_SHA256.startswith("d29d19d7") and NEMO_SHA256.endswith("3e31")
    assert re.fullmatch(r"[0-9a-f]{40}", REVISION)       # a Hub commit, never a branch name
    assert REVISION.startswith("7f32349d")


def test_the_pinned_identity_passes():
    require_identity({"revision": REVISION, "nemo_sha256": NEMO_SHA256})


def test_a_hash_with_only_the_quoted_prefix_and_suffix_is_refused():
    forged = NEMO_SHA256[:8] + "0" * 48 + NEMO_SHA256[-8:]
    with pytest.raises(ScreenStop, match="sha"):
        require_identity({"revision": REVISION, "nemo_sha256": forged})


def test_another_revision_is_refused():
    with pytest.raises(ScreenStop, match="revision"):
        require_identity({"revision": "0" * 40, "nemo_sha256": NEMO_SHA256})


@pytest.mark.parametrize("missing", ["revision", "nemo_sha256"])
def test_an_identity_missing_a_pin_is_refused(missing):
    record = {"revision": REVISION, "nemo_sha256": NEMO_SHA256}
    del record[missing]
    with pytest.raises(ScreenStop, match=missing):
        require_identity(record)


# --- mapping gate -----------------------------------------------------------------------------


def test_exactly_95_percent_located_passes():
    mapping_gate(located=950, total=1000)


def test_just_under_95_percent_located_stops_with_the_counts():
    with pytest.raises(ScreenStop, match="949"):
        mapping_gate(located=949, total=1000)


def test_nothing_to_locate_stops():
    with pytest.raises(ScreenStop):
        mapping_gate(located=0, total=0)


# --- the card sanity line ---------------------------------------------------------------------------


def test_the_cards_ctc_wer_is_7_61_on_a_200_clip_sample():
    assert (CARD_WER, CARD_CLIPS) == (7.61, 200)


def test_a_wer_of_exactly_twice_the_cards_passes():
    card_gate(wer=2 * CARD_WER, clips=CARD_CLIPS)


def test_a_wer_above_twice_the_cards_stops():
    with pytest.raises(ScreenStop, match="WER"):
        card_gate(wer=2 * CARD_WER + 0.01, clips=CARD_CLIPS)


def test_the_card_line_on_another_sample_size_is_refused():
    with pytest.raises(ScreenStop, match="200"):
        card_gate(wer=7.0, clips=199)


# --- the card sanity line's measurement -----------------------------------------------------------


def test_wer_is_pooled_over_reference_words_and_normalised():
    assert wer([("إِنَّ اللَّهَ", "ان الله")])["wer"] == 0.0      # diacritics and ٱ fold away
    pooled = wer([("كتب الله", "كتب"), ("قال", "قال")])
    assert (pooled["clips"], pooled["errors"], pooled["reference_words"]) == (2, 1, 3)
    assert pooled["wer"] == pytest.approx(100 / 3)


def test_the_sample_is_the_first_two_hundred_everyayah_rows_by_key(tmp_path):
    rows = [{"id": f"everyayah_heldout/{n:04d}", "source": "everyayah_heldout",
             "reference_text": "نص", "audio": f"audio/{n}.wav"} for n in range(250)]
    rows += [{"id": f"tlog_holdout/{n}", "source": "tlog_holdout",
              "reference_text": "نص", "audio": f"audio/t{n}.wav"} for n in range(10)]
    manifest = tmp_path / "benchmark.jsonl"
    manifest.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
                        encoding="utf-8")
    chosen = sample_rows(manifest)
    assert len(chosen) == 200 and [row["id"] for row in chosen] == sorted(
        row["id"] for row in chosen)
    assert {row["source"] for row in chosen} == {"everyayah_heldout"}


def test_a_short_sample_stops(tmp_path):
    manifest = tmp_path / "benchmark.jsonl"
    manifest.write_text(json.dumps({"id": "everyayah_heldout/0", "source": "everyayah_heldout",
                                    "reference_text": "نص", "audio": "a.wav"}) + "\n",
                        encoding="utf-8")
    with pytest.raises(ScreenStop, match="card sanity"):
        sample_rows(manifest)


def test_the_record_fails_the_gate_above_twice_the_cards_wer():
    measured = {"clips": CARD_CLIPS, "errors": 0, "reference_words": 100, "wer": 7.0}
    assert record(measured, identity={})["gate_failed"] is False
    assert record({**measured, "wer": 2 * CARD_WER}, identity={})["gate_failed"] is False
    assert record({**measured, "wer": 2 * CARD_WER + 0.01}, identity={})["gate_failed"] is True
    assert record({**measured, "clips": CARD_CLIPS - 1}, identity={})["gate_failed"] is True


# --- the card sample honours the owner's gates --------------------------------------------------


def card_manifest(tmp_path, rows: int = 20):
    manifest = tmp_path / "benchmark.jsonl"
    manifest.write_text("".join(
        json.dumps({"id": f"everyayah_heldout/{n:04d}", "source": "everyayah_heldout",
                    "reference_text": "نص", "audio": f"audio/{n}.wav"}, ensure_ascii=False) + "\n"
        for n in range(rows)), encoding="utf-8")
    return manifest


def card_gates(tmp_path, *, ram_gb: float = 64.0, disk_gb: float = 500.0):
    from tadabur.tlog_mining.give_way import Gates

    return Gates(tmp_path / "PAUSE", free_disk_gb=lambda: disk_gb,
                 available_ram_gb=lambda: ram_gb)


def test_an_existing_pause_file_stops_the_card_line_before_any_decode(tmp_path):
    manifest = card_manifest(tmp_path)
    (tmp_path / "PAUSE").touch()
    seen = []
    with pytest.raises(ScreenStop, match="pause"):
        measure(manifest, lambda path: seen.append(path) or "نص", batch_size=8, sample=20,
                gates=card_gates(tmp_path))
    assert seen == []


def test_a_pause_between_card_batches_stops_before_the_next_batch(tmp_path):
    manifest = card_manifest(tmp_path)
    gates = card_gates(tmp_path)
    seen = []

    def decode(_path):
        seen.append(_path)
        if len(seen) == 8:
            (tmp_path / "PAUSE").touch()
        return "نص"

    with pytest.raises(ScreenStop, match="8 of 20"):
        measure(manifest, decode, batch_size=8, sample=20, gates=gates)
    assert len(seen) == 8


def test_a_low_ram_floor_stops_the_card_line(tmp_path):
    seen = []
    with pytest.raises(ScreenStop, match="ram"):
        measure(card_manifest(tmp_path), lambda path: seen.append(path) or "نص", batch_size=8,
                sample=20, gates=card_gates(tmp_path, ram_gb=1.0))
    assert seen == []


def test_the_card_step_checks_the_gates_before_restoring_the_model(tmp_path, monkeypatch):
    from tadabur.fastconformer_screen import card_line
    from tadabur.fastconformer_screen import decode as module

    (tmp_path / "PAUSE").touch()
    monkeypatch.setattr(module, "load_ctc_model",
                        lambda *a, **k: pytest.fail("the model must not be restored"))
    with pytest.raises(SystemExit) as exit_:
        card_line.main(["--manifest", str(tmp_path / "benchmark.jsonl"),
                        "--out", str(tmp_path / "card_line.json"),
                        "--nemo", str(tmp_path / "m.nemo"), "--revision", "abc",
                        "--pause-file", str(tmp_path / "PAUSE")])
    assert exit_.value.code == 3

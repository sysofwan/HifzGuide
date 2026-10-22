"""The frozen pause-correction manifest (#130) and the artifacts it changes.

The manifest is derived from the guard's own pause decisions and is an input to Ship A's
derived files. Its rows correct only paused ta-marbuta boundaries; every frozen label — the
word sample, the worklist, the clips manifests, the truth sites' ids and indices — stays put.
"""

from __future__ import annotations

import hashlib
import json
import re
from functools import cache
from pathlib import Path

from tadabur import waqf_correction as wc
from tadabur.ship_a_verified import REJECT_PILE_STRATUM
from tadabur.waqf_segments import _uthmani_words

DIR = Path(__file__).resolve().parent
SHIP_A = DIR / "ship_a"
TRUTH_SITES = DIR / "truth_sites"
POOL_CLIPS = DIR / "mining_pool" / "clips.jsonl"
EXTENSION_CLIPS = SHIP_A / "mining" / "clips.jsonl"
SITES = SHIP_A / "sites.jsonl"
FRAME = SHIP_A / "frame.json"
SITES_SUMMARY = SHIP_A / "sites.summary.json"
WORKLIST_SUMMARY = SHIP_A / "worklist.summary.json"
WORD_SAMPLE = SHIP_A / "word_sample.jsonl"
WORKLIST = SHIP_A / "worklist.jsonl"

#: The sites whose paused ta-marbuta reference is corrected: 56 of the 62 h448→ه sites.
#: The six kept are the four the local check did not pause and the two it could not place.
KEPT = {
    "new_audit:tadabur_spk0017_S23_A3_e47c197e_000001.wav#0@58:ت↔ط",
    "new_audit:tadabur_spk0095_S17_A47_67115993_000013.wav#0@50:ت↔ط",
    "new_audit:tadabur_spk0167_S17_A15_6d312abb_000008.wav#0@61:ت↔ط",
    "new_audit:tadabur_spk0412_S5_A74_cb983a4d_000000.wav#0@79:ت↔ط",
    "new_audit:tadabur_spk0485_S4_A66_70923c7f_000087.wav#2@29:ت↔ط",
    "new_audit:tadabur_spk0603_S97_A5_5016ad75_000000.wav#1@49:ت↔ط",
}

#: Frozen by this issue; the correction changes no verdicted site id or index.
WORD_SAMPLE_SHA256 = "073deb66db996b9c638292468378f9923f33893b00d9e7c04fafb9414ed15e03"
WORKLIST_SHA256 = "097181fd5d56316c057c5f615812e262ab73bcf4a788d6f5e75ef67cd3576bb0"
POOL_CLIPS_SHA256 = "885901089d5ed50f21f40405d0c8568a79fbe2319560b906ea429f46e629cf6e"
EXTENSION_CLIPS_SHA256 = "32f9f4e9dc50344c35ba846c5d9104ebd3c164a0dd435dc9cd26e3d2c9008cf1"
WAQF_TRUTH_SITES_SHA256 = "719144f8b9e7c7b490a50796a17085e1ac5830b0f89ad447d5d2dff54d59cf19"
P35_TRUTH_SITES_SHA256 = "654f8ff890e8dd66ddc4db52522c07267ba85f1b0cdb392f0d7990b86689a2f8"
VERIFIED_ID_INDEX_SHA256 = "6c44464a28b0369f13d8a3846ef420bfa92c52a5287b86b0aa9c51c060f7405b"


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(raw) for raw in f if raw.strip()]


def test_the_committed_manifest_is_the_set_the_guard_derives():
    assert wc.CORRECTION_PATH == SHIP_A / "waqf_correction.jsonl"
    rows = wc.load_corrections()
    assert len(rows) == 56
    assert rows == wc.derive_corrections()


def test_every_row_names_a_clip_a_segment_a_paused_taa_marbuta_cut_and_provenance():
    rows = wc.load_corrections()
    keys = []
    for row in rows:
        words = _uthmani_words(row.surah_ayah)
        assert 0 < row.cut_word < len(words)
        assert "ة" in words[row.cut_word - 1]
        assert row.start_sample >= 0
        assert row.provenance
        keys.append((row.audio_filename, row.start_sample, row.cut_word))
    assert keys == sorted(keys)
    assert len(set(keys)) == len(keys)


def test_the_committed_sites_keep_only_the_unpaused_taa_sites():
    sites = read_jsonl(SITES)
    still_wasl = {
        row["site_id"] for row in sites
        if row["stratum"] == "ship_a:ت:h448_other" and row["h448_committed"] == "ه"
    }
    assert still_wasl == KEPT


def test_verified_sites_keep_their_ids_and_reference_indices():
    sites = {row["site_id"]: row for row in read_jsonl(SITES)}
    verified = read_jsonl(TRUTH_SITES / "ship_a_verified.jsonl")
    pairs = sorted((row["site_id"], row["reference_index"]) for row in verified)
    digest = hashlib.sha256(
        json.dumps(pairs, ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    assert digest == VERIFIED_ID_INDEX_SHA256
    # #87's reject-pile draws are no Ship A site (tadabur.ship_a_verified), so they are not
    # in sites.jsonl; their ids stay as the truth file holds them.
    reject_pile = [row for row in verified if row["stratum"].endswith(REJECT_PILE_STRATUM)]
    assert len(reject_pile) == 15
    assert all("↔" in row["mark"] for row in reject_pile)
    assert not any(row["site_id"] in sites for row in reject_pile)
    ship_a = [row for row in verified if not row["stratum"].endswith(REJECT_PILE_STRATUM)]
    assert len(ship_a) == 515
    for row in ship_a:
        assert row["site_id"] in sites
        assert sites[row["site_id"]]["reference_index"] == row["reference_index"]


def test_worklist_sites_keep_their_reference_indices():
    sites = {row["site_id"]: row for row in read_jsonl(SITES)}
    for row in read_jsonl(WORKLIST):
        if row["site_id"] in sites:
            assert sites[row["site_id"]]["reference_index"] == row["reference_index"]


def test_the_word_sample_is_byte_identical():
    assert sha256_of(WORD_SAMPLE) == WORD_SAMPLE_SHA256


def test_the_worklist_is_byte_identical():
    assert sha256_of(WORKLIST) == WORKLIST_SHA256


def test_the_clips_manifests_are_untouched():
    assert sha256_of(POOL_CLIPS) == POOL_CLIPS_SHA256
    assert sha256_of(EXTENSION_CLIPS) == EXTENSION_CLIPS_SHA256


def test_the_waqf_and_p35_truth_sites_are_untouched():
    assert sha256_of(TRUTH_SITES / "waqf_boundaries.jsonl") == WAQF_TRUTH_SITES_SHA256
    assert sha256_of(TRUTH_SITES / "p35_fixtures.jsonl") == P35_TRUTH_SITES_SHA256


def test_the_manifest_is_an_input_of_the_frame_and_the_sites_summary():
    digest = sha256_of(wc.CORRECTION_PATH)
    frame = json.loads(FRAME.read_text(encoding="utf-8"))
    summary = json.loads(SITES_SUMMARY.read_text(encoding="utf-8"))
    assert frame["inputs"]["ship_a/waqf_correction.jsonl"] == digest
    assert summary["inputs"]["ship_a/waqf_correction.jsonl"] == digest


def test_the_worklist_summary_freezes_the_current_sites():
    summary = json.loads(WORKLIST_SUMMARY.read_text(encoding="utf-8"))
    assert summary["inputs"]["ship_a/sites.jsonl"] == sha256_of(SITES)


@cache
def _words(surah_ayah: str) -> tuple[str, ...]:
    return tuple(_uthmani_words(surah_ayah))


def _kept_segments(path: Path) -> tuple[int, int]:
    """(kept segments, kept segments whose last word is a taa marbuta read in wasl)."""
    kept = wasl_taa = 0
    for clip in read_jsonl(path):
        words = _words(clip["surah_ayah"])
        for segment in clip["segments"]:
            if not segment["kept"]:
                continue
            kept += 1
            if "ة" not in words[segment["word_end"] - 1]:
                continue
            if re.search("ت[َُِ](?:وو|ںں|ۥۥ|و|ں)$", segment["reference"].rstrip()):
                wasl_taa += 1
    return kept, wasl_taa


def test_no_kept_segment_ends_on_a_wasl_taa_marbuta():
    assert _kept_segments(EXTENSION_CLIPS) == (2016, 0)
    assert _kept_segments(POOL_CLIPS) == (3785, 0)

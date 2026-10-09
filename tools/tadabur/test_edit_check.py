"""Tests for the synthetic-edit check in the listening session (#107): loading the blind
check with its words, fetching its audio beside the session's clips, and the per-operation
summary #90 reads. The page side is in ``test_tashkeel_audit_ui.py``."""

from __future__ import annotations

import json
import shutil
from dataclasses import replace

import numpy as np
import pytest
import soundfile as sf

from tadabur.edit_check import (
    BLIND_CHECK_PATH,
    EditCheckSite,
    load_edit_check,
    read_words,
    verify_audio,
)
from tadabur.edit_check_summary import summarize
from tadabur.listening_session import (
    Candidate,
    Verdict,
    census,
    fetch,
)
from tadabur.staged_audio import StagedClip
from tadabur.synthetic_edit_plan import CONSONANT_SWAP, DECOY, EDIT, SHADDAH_REMOVED
from tadabur.synthetic_edits import WORKLIST_PATH, read_items
from tadabur.truth_sites import PENDING, TruthSite, audio_sha256, load_truth_sites

FATHA = "َ"
#: ب doubled at 2 and ذ at 5: one shaddah and one consonant carrier.
REFERENCE = f"ر{FATHA}بب{FATHA}ذ{FATHA}ب"


def _site(number: int, index: int, mark: str, prescribed: str, sha: str = "a" * 64,
          reference: str = REFERENCE) -> TruthSite:
    return TruthSite(
        site_id=f"synthetic_edit:{number:020x}", source="synthetic_edit",
        assumes_competent_reciter=False, audio_filename=f"se_{number:020x}.wav", shard=200,
        start_sample=0, end_sample=16000, audio_sha256=sha, surah_ayah="2:2",
        reference=reference, reference_index=index, mark=mark, prescribed=prescribed,
        heard=PENDING, stratum="synthetic_edit:blind_check", stratum_population=10)


def _write(tmp_path, sites, words=None):
    from tadabur.truth_sites import write_truth_sites

    write_truth_sites(sites, tmp_path / "blind_check.jsonl")
    words = words or {"2:2": {"reference": REFERENCE, "word_offsets": [0, 5, len(REFERENCE)]}}
    (tmp_path / "words.json").write_text(json.dumps({"ayahs": words}), encoding="utf-8")
    return tmp_path / "blind_check.jsonl", tmp_path / "words.json"


# --- loading ---------------------------------------------------------------------------


def test_the_committed_blind_check_loads_with_its_words():
    assert BLIND_CHECK_PATH == WORKLIST_PATH
    rows = load_edit_check()
    assert len(rows) == len(load_truth_sites(WORKLIST_PATH))
    for row in rows:
        assert (row.excerpt_start_sample, row.excerpt_end_sample) == (0, row.site.end_sample)
        assert row.mode in ("shaddah", "consonant") and row.word_start == 0
        assert row.word_offsets[0] <= row.site.reference_index < row.word_offsets[-1]


def test_an_item_whose_reference_has_no_words_is_refused(tmp_path):
    other = REFERENCE + "ب"
    paths = _write(tmp_path, [_site(0, 2, "shaddah", "held", reference=other)])
    with pytest.raises(ValueError, match="tadabur.edit_check words"):
        load_edit_check(*paths)


def test_words_that_do_not_partition_the_reference_are_refused(tmp_path):
    paths = _write(tmp_path, [_site(0, 2, "shaddah", "held")],
                   {"2:2": {"reference": REFERENCE, "word_offsets": [0, 5]}})
    with pytest.raises(ValueError, match="partition"):
        read_words(paths[1])


def test_only_pending_whole_synthetic_edit_items_are_asked(tmp_path):
    with pytest.raises(ValueError, match="pending synthetic-edit"):
        load_edit_check(*_write(tmp_path, [replace(_site(0, 2, "shaddah", "held"),
                                                    heard="held")]))
    with pytest.raises(ValueError, match="whole"):
        load_edit_check(*_write(tmp_path, [replace(_site(0, 2, "shaddah", "held"),
                                                    start_sample=10)]))


def test_audio_is_verified_by_checksum_and_a_missing_item_says_how_to_fetch_it(tmp_path):
    path = tmp_path / "se_00000000000000000000.wav"
    sf.write(path, np.full(16000, 0.1, dtype=np.float32), 16000, subtype="PCM_16")
    row = EditCheckSite(_site(0, 2, "shaddah", "held", sha=audio_sha256(path)), (0, 13))
    verify_audio([row], tmp_path)
    with pytest.raises(ValueError, match="sha256"):
        verify_audio([replace(row, site=replace(row.site, audio_sha256="b" * 64))], tmp_path)
    path.unlink()
    with pytest.raises(FileNotFoundError, match="listening_session fetch"):
        verify_audio([row], tmp_path)


# --- fetching --------------------------------------------------------------------------


def test_fetch_copies_both_kinds_with_one_rsync_and_verifies_every_file(tmp_path):
    remote = tmp_path / "remote"
    (remote / "clips").mkdir(parents=True)
    (remote / "edits").mkdir()
    sf.write(remote / "clips" / "c.wav", np.full(32000, 0.2, dtype=np.float32), 16000,
             subtype="PCM_16")
    sf.write(remote / "edits" / "se_00000000000000000000.wav",
             np.full(16000, 0.1, dtype=np.float32), 16000, subtype="PCM_16")
    clip = StagedClip("c.wav", 40, 1, 7, "2:2", 32000, audio_sha256(remote / "clips" / "c.wav"),
                      ("mining_pool",))
    site = replace(_site(0, 2, "shaddah", "held"), source="new_audit", stratum="new_audit:s",
                   audio_filename="c.wav", audio_sha256=clip.audio_sha256, end_sample=32000,
                   site_id="new_audit:c")
    rows = census([Candidate(site, 1.0, 0, (0, len(REFERENCE)), (0, 16000))])
    item = EditCheckSite(_site(0, 2, "shaddah", "held", sha=audio_sha256(
        remote / "edits" / "se_00000000000000000000.wav")), (0, len(REFERENCE)))
    calls = []

    def rsync(command, input, **kwargs):
        calls.append((command, input))
        source_root, local = command[-2].split(":", 1)[1], command[-1]
        for line in input.splitlines():
            shutil.copy(f"{source_root}{line}", local)

    local = tmp_path / "local"
    fetch(rows, [item], local, {"c.wav": clip}, host="box",
          sources={"session": str(remote / "clips"), "edit_check": str(remote / "edits")},
          run=rsync)
    ((command, listed),) = calls
    assert command[:4] == ["rsync", "-a", "--no-relative", "--files-from=-"]
    assert command[-2] == f"box:{remote}/" and listed.splitlines() == [
        "clips/c.wav", "edits/se_00000000000000000000.wav"]
    assert sorted(p.name for p in local.iterdir()) == ["c.wav", "se_00000000000000000000.wav"]
    with pytest.raises(ValueError):  # a copy that does not match is never accepted
        fetch(rows, [replace(item, site=replace(item.site, audio_sha256="b" * 64))], local,
              {"c.wav": clip}, host="box", run=lambda *a, **k: None)


# --- the summary for #90 ---------------------------------------------------------------


def _manifest(sites_and_roles):
    """Minimal manifest rows for blind-check sites: the fields the summary reads."""
    items = []
    for site, operation, role, label in sites_and_roles:
        items.append({
            "operation": operation, "mark": site.mark, "role": role, "label": label,
            "reference": site.reference, "reference_index": site.reference_index,
            "prescribed": site.prescribed,
            "output": {"audio_filename": site.audio_filename, "audio_sha256": site.audio_sha256},
        })
    return items


def test_the_summary_counts_naturalness_and_labels_per_operation_and_role():
    removed_edit = _site(0, 2, "shaddah", "held")
    removed_decoy = _site(1, 2, "shaddah", "held")
    swap_edit = _site(2, 5, "ذ↔ظ", "ذ")
    swap_decoy = _site(3, 5, "ذ↔ظ", "ذ")
    unanswered = _site(4, 2, "shaddah", "held")
    items = _manifest([(removed_edit, SHADDAH_REMOVED, EDIT, "not_held"),
                       (removed_decoy, SHADDAH_REMOVED, DECOY, "held"),
                       (swap_edit, CONSONANT_SWAP, EDIT, "ظ"),
                       (swap_decoy, CONSONANT_SWAP, DECOY, "ذ"),
                       (unanswered, SHADDAH_REMOVED, EDIT, "not_held")])
    verdicts = {s.site_id: Verdict(s.site_id, heard, "", natural) for s, heard, natural in [
        (removed_edit, "not_held", "natural"), (removed_decoy, "held", "unclear"),
        (swap_edit, "unclear", "unnatural"), (swap_decoy, "ظ", "natural")]}
    summary = summarize([removed_edit, removed_decoy, swap_edit, swap_decoy, unanswered],
                        items, verdicts)
    assert (summary["items"], summary["answered"]) == (5, 4)
    removed = summary["operations"][SHADDAH_REMOVED]
    assert removed[EDIT] == {
        "items": 2, "pending": 1, "natural": 1, "unnatural": 0, "natural_unclear": 0,
        "natural_rate": 1.0, "heard_as_labelled": 1, "heard_otherwise": 0,
        "heard_unclear": 0, "heard_as_labelled_rate": 1.0}
    assert removed[DECOY]["natural_rate"] is None  # only "unclear": no denominator
    assert removed[DECOY]["heard_as_labelled_rate"] == 1.0  # the decoy heard unchanged
    swap = summary["operation_marks"]["consonant_swap ذ↔ظ"]
    assert (swap[EDIT]["natural_rate"], swap[EDIT]["heard_as_labelled_rate"]) == (0.0, None)
    assert swap[DECOY]["heard_as_labelled_rate"] == 0.0  # the decoy heard as changed


def test_the_summary_refuses_a_verdict_without_naturalness_or_a_mismatched_manifest():
    site = _site(0, 2, "shaddah", "held")
    items = _manifest([(site, SHADDAH_REMOVED, EDIT, "not_held")])
    with pytest.raises(ValueError, match="both questions"):
        summarize([site], items, {site.site_id: Verdict(site.site_id, "held")})
    items[0]["reference_index"] = 5
    with pytest.raises(ValueError, match="disagree"):
        summarize([site], items, {})


def test_the_committed_blind_check_joins_the_committed_manifest():
    summary = summarize(load_truth_sites(WORKLIST_PATH), read_items(), {})
    assert summary["answered"] == 0
    assert sum(cell["items"] for roles in summary["operations"].values()
               for cell in roles.values()) == summary["items"]

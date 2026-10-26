"""S5b (#204): the fresh confirmation draws, the five bars, the oracle and the commit order.

Tests-first: this file pins the contract the implementation must satisfy, written before any of it.
The draws are pure functions of the committed manifests, the salt and the cap, so every selection
is re-derived here from small synthetic inputs plus the committed files. No test downloads anything.

The rule (issue #204): a decode with speech is ``non_recitation`` when its normalised length is
under ``L_MIN`` or its own decoy best is within ``DELTA`` of its real best. ``L_MIN`` and ``DELTA``
are fixed in ``docs/tlog-ident-floor-preregistration.md`` and never selected on any data here.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import random
import subprocess
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FLOOR_PREREGISTRATION = ROOT / "docs" / "tlog-ident-floor-preregistration.md"
IDENT_DIR = ROOT / "tools" / "tadabur" / "tlog" / "ident"
CONFIRM_DIR = IDENT_DIR / "confirm"
FREESOUND_MANIFEST = CONFIRM_DIR / "freesound.json"
CV_MANIFEST = CONFIRM_DIR / "common_voice.jsonl"
CV_COUNTS = CONFIRM_DIR / "common_voice.json"
NOISE_POOL = ROOT / "tools" / "tadabur" / "nonpro_gap" / "noise_pool.json"
AUGMENTATION_POOL = ROOT / "tools" / "tadabur" / "augmentation" / "noise_pool.json"
IMLAEI = ROOT / "data" / "imlaei.json"

#: The frozen parents the floor re-pre-registration cites by SHA-256 (never edited).
HOLDOUT_SHA256 = "204fe7f980488aff73402c56f1729f7f17f7c9471ea0664f6365bb8870b62086"
S5_PREREGISTRATION_SHA256 = (
    "a383fcfdbd375c9e6c052d8032788a8eda69bd2d9fbd12fcd3b921b1fbb6a3ab")
BU_AMENDMENT_SHA256 = "579f9ba1bb751e65663e3a6f477b6d39c8503c94811050951e1338cc71aa76c8"
#: The six primary MDC datasets the owner accepted (issue #204 comment, 2026-10-26).
MDC_DATASETS = {
    "ar": "cmu5wom3z00ceo107guxtg4xa",
    "en": "cmu5jplf300nwmh07iqvk9leo",
    "ur": "cmu5x1w3400cyo107st8cgsoq",
    "fa": "cmu5xazgy00finq074vaqo1b3",
    "tr": "cmu5wkah500c2o10719lem8m8",
    "id": "cmu5wcato00cknq07yjnu14e0",
}
AUDIO_SUFFIXES = (".mp3", ".wav", ".flac", ".ogg", ".opus", ".m4a", ".aac", ".webm")


@pytest.fixture(scope="session")
def ident():
    return importlib.import_module("tadabur.tlog_ident")


@pytest.fixture(scope="session")
def confirm():
    return importlib.import_module("tadabur.tlog_ident_confirm")


@pytest.fixture(scope="session")
def calibration(ident):
    return json.loads(ident.CALIBRATION_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cv_rank(salt: str, locale: str, path: str) -> bytes:
    return hashlib.sha256(f"{salt}:cv:{locale}:{path}".encode("utf-8")).digest()


def _freesound_rank(salt: str, freesound_id: int) -> bytes:
    return hashlib.sha256(f"{salt}:fs:{freesound_id}".encode("utf-8")).digest()


def _shard_rank(salt: str, shard: str) -> bytes:
    return hashlib.sha256(f"{salt}:shard:{shard}".encode("utf-8")).digest()


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True).stdout


def _first_commit(path: Path) -> str:
    """The commit that added ``path`` (the oldest one)."""
    out = _git("log", "--format=%H", "--diff-filter=A", "--",
               str(path.relative_to(ROOT))).split()
    assert out, f"{path} has no adding commit"
    return out[-1]


def _last_commit(path: Path) -> str:
    out = _git("log", "-1", "--format=%H", "--", str(path.relative_to(ROOT))).split()
    assert out, f"{path} has no commit"
    return out[0]


def _is_ancestor(earlier: str, later: str) -> bool:
    return subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor",
                           earlier, later], capture_output=True).returncode == 0


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _clip(path: str, client: str) -> dict:
    return {"path": path, "client_id": client, "sentence": "x"}


# --- the pre-registration: parameters, doc pin and parents -------------------------------------


def test_the_confirm_parameters_are_fixed_before_any_draw(confirm):
    assert confirm.SALT == "tlog-s5b-floor-confirm-v1"
    assert (confirm.FREESOUND_SIZE, confirm.CV_AR_SIZE) == (1000, 500)
    assert confirm.CV_OTHER_PER_LOCALE == 100
    assert confirm.PRIMARY_LOCALES == ("en", "ur", "fa", "tr", "id")
    assert confirm.SUBSTITUTES == {"en": "es", "ur": "hi", "fa": "ckb", "tr": "sw", "id": "bn"}
    assert confirm.ARABIC_SCRIPT == frozenset({"ar", "ur", "fa", "ckb"})
    assert {confirm.shard_index(name) for name in confirm.candidate_shards()} == set(
        range(104)) - {0, 26, 52, 78}
    assert confirm.MAX_SHARDS == 10
    assert confirm.CV_RELEASE == "cv-corpus-27.0-2026-09-11"


def test_the_floor_preregistration_is_committed_and_cites_its_frozen_parents(ident, confirm):
    """The doc is dated, post-look, cites the three frozen parents by SHA-256 and pins every branch
    (rule, sizes, salt, speaker cap, substitutes, the release and the archive hashes)."""
    assert FLOOR_PREREGISTRATION.is_file()
    sha = _sha256(FLOOR_PREREGISTRATION)
    assert sha == ident.FLOOR_PREREGISTRATION_SHA256
    assert ident.floor_rule() == {"path": "docs/tlog-ident-floor-preregistration.md",
                                  "sha256": sha}
    assert (ident.L_MIN, ident.DELTA) == (8, pytest.approx(0.30))
    assert ident.FLOOR_SALT == confirm.SALT

    text = FLOOR_PREREGISTRATION.read_text(encoding="utf-8")
    assert "post-look" in text and "2026-10-26" in text
    for parent in (HOLDOUT_SHA256, S5_PREREGISTRATION_SHA256, BU_AMENDMENT_SHA256):
        assert parent in text
    assert confirm.SALT in text
    assert confirm.CV_RELEASE in text
    assert f"L_min = {ident.L_MIN}" in text and "0.30" in text
    assert "1,000" in text or "1000" in text
    for locale, dataset in MDC_DATASETS.items():
        assert dataset in text, f"the doc does not pin the {locale} archive"
    for primary, substitute in confirm.SUBSTITUTES.items():
        assert substitute in text
    assert "cap" in text.lower()


# --- the draw rules ----------------------------------------------------------------------------


def test_the_salted_rank_is_deterministic_and_matches_the_pinned_formula(confirm):
    assert confirm.salted_rank("cv", "ar", "a.mp3") == _cv_rank(confirm.SALT, "ar", "a.mp3")
    assert confirm.salted_rank("fs", 698) == _freesound_rank(confirm.SALT, 698)
    assert confirm.salted_rank("shard", "s") == _shard_rank(confirm.SALT, "s")
    assert confirm.salted_rank("cv", "ar", "a.mp3") == confirm.salted_rank("cv", "ar", "a.mp3")
    assert confirm.salted_rank("cv", "ar", "a.mp3") != confirm.salted_rank("cv", "ar", "b.mp3")


def test_the_shard_ranking_is_the_salted_order_of_the_100_shared_shards(confirm):
    ranking = confirm.shard_ranking(confirm.candidate_shards())
    assert len(ranking) == len(set(ranking)) == 100
    assert set(ranking) == set(confirm.candidate_shards())
    expected = sorted(ranking, key=lambda shard: _shard_rank(confirm.SALT, shard))
    assert list(ranking) == expected
    assert not {shard for shard in ranking
                if confirm.shard_index(shard) in confirm.HELD_OUT_SHARDS}


def test_the_freesound_selection_is_the_first_1000_by_salted_rank(confirm):
    eligible = [{"freesound_id": index} for index in range(1200)]
    selected = confirm.freesound_select(eligible)
    expected = sorted(eligible, key=lambda row: _freesound_rank(
        confirm.SALT, row["freesound_id"]))[:confirm.FREESOUND_SIZE]
    assert [row["freesound_id"] for row in selected] == [
        row["freesound_id"] for row in expected]
    assert len(selected) == 1000
    assert len(confirm.freesound_select(eligible[:37], count=100)) == 37


def test_the_cv_draw_is_pass_1_at_most_one_clip_per_speaker_then_pass_2_at_two(confirm):
    # Plenty of speakers: pass 1 fills the quota, cap 1.
    many = [_clip(f"c{index}.mp3", f"spk{index}") for index in range(200)]
    filled = confirm.cv_draw(many, locale="en", count=100)
    assert len(filled["drawn"]) == 100
    assert filled["cap_reached"] == 1
    assert len({row["client_id"] for row in filled["drawn"]}) == 100

    # 60 speakers, 200 clips: pass 1 takes 60, pass 2 fills to 120 -> takes 100 at cap 2.
    clustered = [_clip(f"c{index}.mp3", f"spk{index % 60}") for index in range(200)]
    escalated = confirm.cv_draw(clustered, locale="ur", count=100)
    assert len(escalated["drawn"]) == 100
    assert escalated["cap_reached"] == 2
    counts = Counter(row["client_id"] for row in escalated["drawn"])
    assert max(counts.values()) == 2
    # Every speaker's first clip is in before any speaker's second: the same ranking.
    ranking = {row["path"]: _cv_rank(confirm.SALT, "ur", row["path"]) for row in clustered}
    taken = [ranking[row["path"]] for row in escalated["drawn"]]
    assert taken == sorted(taken)

    # 20 speakers: pass 2 reaches 40, short of 100; the draw stops and reports the n.
    short = [_clip(f"c{index}.mp3", f"spk{index % 20}") for index in range(120)]
    starved = confirm.cv_draw(short, locale="id", count=100)
    assert len(starved["drawn"]) == 40
    assert starved["cap_reached"] is None
    assert starved["short"] == 60


def test_the_draw_is_one_rule_so_both_fetch_routes_agree(confirm):
    """The full-archive and two-pass routes differ only in how bytes arrive; the draw is a pure
    function of the rows, so the manifest is identical whichever order the rows are read in."""
    rows = [_clip(f"c{index}.mp3", f"spk{index % 90}") for index in range(300)]
    first = confirm.cv_draw(rows, locale="en", count=100)
    shuffled = list(rows)
    random.Random(20261026).shuffle(shuffled)
    second = confirm.cv_draw(shuffled, locale="en", count=100)
    third = confirm.cv_draw(reversed(rows), locale="en", count=100)
    assert first == second == third


def test_the_substitute_fires_only_below_100_eligible_at_cap_2(confirm):
    viable = {locale: 100 for locale in confirm.PRIMARY_LOCALES}
    planned = confirm.locale_plan(viable)
    assert {locale: entry["locale"] for locale, entry in planned.items()} == {
        locale: locale for locale in confirm.PRIMARY_LOCALES}
    assert not any(entry["substituted"] for entry in planned.values())

    starved = {**viable, "ur": 99, "hi": 100}
    planned = confirm.locale_plan(starved)
    assert planned["ur"] == {"locale": "hi", "substituted": True}
    assert planned["en"] == {"locale": "en", "substituted": False}

    both_short = {**viable, "ur": 99, "hi": 99}
    with pytest.raises(confirm.ConfirmDrawError, match="ur"):
        confirm.locale_plan(both_short)

    missing = {locale: 100 for locale in confirm.PRIMARY_LOCALES if locale != "fa"}
    with pytest.raises(confirm.ConfirmDrawError, match="fa"):
        confirm.locale_plan(missing)


# --- the Quran-text guard ----------------------------------------------------------------------


def test_the_guard_tokens_strip_tashkeel_tatweel_punctuation_and_digits(confirm):
    assert confirm.guard_tokens("بِسْمِ اللَّهِ، ١٢٣") == ("بسم", "الله")
    assert confirm.guard_tokens("بِسْـــمِ") == ("بسم",)
    assert confirm.guard_tokens("  ") == ()


def _imlaei(words: list[tuple[int, int, int, str]]) -> dict:
    return {f"{s}:{a}:{w}": {"surah": str(s), "ayah": str(a), "word": str(w), "text": text}
            for s, a, w, text in words}


def test_the_guard_catches_a_four_word_quote_across_an_ayah_boundary(confirm):
    words = [(1, 1, 1, "بسم"), (1, 1, 2, "الله"), (1, 2, 1, "الرحمن"),
             (1, 2, 2, "الرحيم"), (1, 2, 3, "مالك")]
    ngrams = confirm.quran_ngrams(_imlaei(words))
    # Consecutive words per SURAH, so a quote spanning an ayah boundary is caught.
    assert ("بسم", "الله", "الرحمن", "الرحيم") in ngrams
    assert ("الله", "الرحمن", "الرحيم", "مالك") in ngrams
    guard = lambda sentence: confirm.fails_quran_guard(sentence, ngrams)  # noqa: E731
    # The sentence is normalised too: its tashkeel must not hide the quote.
    assert guard("قال: بِسْمِ اللَّهِ الرَّحْمٰنِ الرَّحِيمِ، ثم سكت")
    assert guard("اللَّهِ الرَّحْمٰنِ الرَّحِيمِ مَالِكِ")
    # Three words are not a four-word run, and an unrelated sentence is clean.
    assert not guard("بِسْمِ اللَّهِ الرَّحْمٰنِ")
    assert not guard("السلام عليكم ورحمة الله وبركاته")


def test_no_committed_cv_sentence_is_a_quran_quote(confirm):
    assert CV_MANIFEST.is_file(), "the CV manifest is not committed yet"
    ngrams = confirm.quran_ngrams(json.loads(IMLAEI.read_text(encoding="utf-8")))
    rows = _jsonl(CV_MANIFEST)
    failing = [row["path"] for row in rows if confirm.fails_quran_guard(row["sentence"], ngrams)]
    assert failing == []


# --- the committed manifests -------------------------------------------------------------------


def test_the_freesound_manifest_rederives_from_its_shard_list_and_salt(confirm):
    manifest = json.loads(FREESOUND_MANIFEST.read_text(encoding="utf-8"))
    assert manifest["salt"] == confirm.SALT
    assert manifest["dataset"]["repo"] == "obadx/freesound-commercial-50k-noise-only"
    assert manifest["dataset"]["revision"] == "a5f48ab83b64ba1d35ee4ec363dae75d4e2c69d3"

    candidates = confirm.candidate_shards()
    assert manifest["dataset"]["shards"] == list(confirm.shard_ranking(candidates))
    drawn = manifest["drawn_shards"]
    assert drawn == manifest["dataset"]["shards"][:len(drawn)]
    assert len(drawn) <= confirm.MAX_SHARDS
    assert not {confirm.shard_index(shard) for shard in drawn} & confirm.HELD_OUT_SHARDS

    eligible = manifest["eligible"]
    assert {row["shard"] for row in eligible} <= set(drawn)
    recordings = manifest["recordings"]
    expected = sorted(eligible, key=lambda row: _freesound_rank(
        confirm.SALT, row["freesound_id"]))[:confirm.FREESOUND_SIZE]
    assert recordings == expected
    assert len(recordings) == len({row["freesound_id"] for row in recordings})
    if len(recordings) < confirm.FREESOUND_SIZE:
        assert len(drawn) == confirm.MAX_SHARDS  # the 10-shard stop, reported not redrawn
        assert manifest["short"] == confirm.FREESOUND_SIZE - len(recordings)
    else:
        assert len(recordings) == confirm.FREESOUND_SIZE

    for row in recordings + eligible:
        assert set(row) == {"freesound_id", "licence", "username", "samples_sha256",
                            "seconds", "shard"}
        assert len(row["samples_sha256"]) == 64
        assert row["seconds"] >= 1.0
    assert manifest["licences"] == dict(Counter(row["licence"] for row in recordings))
    assert sum(manifest["licences"].values()) == len(recordings)
    for shard in drawn:
        left = manifest["left_out"][shard]
        assert set(left) == {"undecodable", "ineligible", "in_nonpro_gap_pool",
                             "in_augmentation_pool"}
        assert all(value >= 0 for value in left.values())


def test_the_freesound_draw_shares_no_id_or_sample_with_either_committed_pool(confirm):
    manifest = json.loads(FREESOUND_MANIFEST.read_text(encoding="utf-8"))
    ids, samples = set(), set()
    for path in (NOISE_POOL, AUGMENTATION_POOL):
        for record in json.loads(path.read_text(encoding="utf-8"))["recordings"]:
            ids.add(record["freesound_id"])
            samples.add(record["samples_sha256"])
    assert not {row["freesound_id"] for row in manifest["recordings"]} & ids
    assert not {row["samples_sha256"] for row in manifest["recordings"]} & samples
    assert not {row["freesound_id"] for row in manifest["eligible"]} & ids
    assert not {row["samples_sha256"] for row in manifest["eligible"]} & samples


def test_the_cv_manifest_is_500_arabic_plus_500_non_arabic_with_the_cap_respected(confirm):
    assert CV_COUNTS.is_file(), "the CV counts sidecar is not committed yet"
    rows = _jsonl(CV_MANIFEST)
    counts = json.loads(CV_COUNTS.read_text(encoding="utf-8"))
    assert counts["salt"] == confirm.SALT
    assert counts["release"] == confirm.CV_RELEASE
    assert len(counts["archives"]) == 11  # the six primaries and the five substitutes
    for archive in counts["archives"]:
        assert set(archive) == {"locale", "dataset_id", "filename", "sha256", "size_bytes"}
        assert len(archive["sha256"]) == 64
        assert archive["size_bytes"] > 0
    assert {archive["dataset_id"] for archive in counts["archives"]} >= set(
        MDC_DATASETS.values())

    assert sum(row["arm"] == "ar" for row in rows) == confirm.CV_AR_SIZE == 500
    assert sum(row["arm"] == "other" for row in rows) == 500
    per_locale = Counter(row["locale"] for row in rows)
    assert per_locale["ar"] == confirm.CV_AR_SIZE
    for row in rows:
        assert set(row) == {"arm", "locale", "path", "client_id", "sentence", "audio_sha256",
                            "duration_ms", "release"}
        assert len(row["audio_sha256"]) == 64
        assert row["duration_ms"] > 0
        assert row["release"] == confirm.CV_RELEASE
        assert row["path"] == Path(row["path"]).name  # a bare clip name, not a repo path
        assert row["arm"] == ("ar" if row["locale"] == "ar" else "other")

    # The counts are keyed by the primary arm; ``locale`` is what was actually drawn (a
    # substitute only when the primary could not reach 100 at cap 2).
    assert set(counts["locales"]) == {"ar", *confirm.PRIMARY_LOCALES}
    for primary, entry in counts["locales"].items():
        assert set(entry) == {"primary", "locale", "substituted", "cap_reached", "eligible",
                              "excluded_guard", "examined"}
        assert entry["primary"] == primary
        assert entry["cap_reached"] in (1, 2) or entry["cap_reached"] is None
        assert per_locale[entry["locale"]] == 100
        counts_by_speaker = Counter(row["client_id"] for row in rows
                                    if row["locale"] == entry["locale"])
        if entry["cap_reached"] is not None:
            assert max(counts_by_speaker.values()) <= entry["cap_reached"]
        if entry["locale"] != primary:
            assert entry["substituted"] is True
            assert entry["locale"] == confirm.SUBSTITUTES[primary]
        else:
            assert entry["substituted"] is False


def test_the_cv_manifest_is_tracked_and_no_audio_is_committed():
    tracked = _git("ls-files", "tools/tadabur/tlog/ident/confirm").splitlines()
    assert "tools/tadabur/tlog/ident/confirm/common_voice.jsonl" in tracked
    assert not [name for name in tracked if name.lower().endswith(AUDIO_SUFFIXES)]


# --- the five bars -----------------------------------------------------------------------------


def test_the_five_bars_use_the_limits_wilson_intervals_and_the_first_fired_bar(confirm):
    def negatives(passed: int, total: int) -> list[bool]:
        return [True] * passed + [False] * (total - passed)

    passing = confirm.five_bars(
        freesound=negatives(10, 1000), cv_ar=negatives(5, 500), cv_other=negatives(5, 500),
        mistakes=[False] * 52, look_retention=0.90)
    assert passing["status"] == "confirmed"
    assert passing["freesound"]["limit"] == pytest.approx(0.01)
    assert passing["freesound"]["clips"] == 1000
    assert passing["freesound"]["passed"] == 10
    assert passing["freesound"]["fired"] is False  # at the limit is not over it
    assert passing["look_retention"]["fired"] is False
    assert passing["mistakes"] == {"clips": 52, "passed": 52, "fired": False}
    for arm in ("freesound", "cv_ar", "cv_other"):
        low, high = passing[arm]["wilson"]
        assert 0.0 <= low <= passing[arm]["pass"] <= high <= 1.0

    over = confirm.five_bars(
        freesound=negatives(11, 1000), cv_ar=negatives(0, 500), cv_other=negatives(0, 500),
        mistakes=[False] * 52, look_retention=0.99)
    assert over["status"] == "stopped: freesound 0.011 > 0.01"
    assert over["freesound"]["fired"] is True

    missed = confirm.five_bars(
        freesound=negatives(0, 1000), cv_ar=negatives(0, 500), cv_other=negatives(0, 500),
        mistakes=[True] * 52, look_retention=0.99)
    assert missed["status"] == "stopped: mistakes 0/52"
    assert missed["mistakes"]["fired"] is True

    dropped = confirm.five_bars(
        freesound=negatives(0, 1000), cv_ar=negatives(0, 500), cv_other=negatives(0, 500),
        mistakes=[False] * 52, look_retention=0.899)
    assert dropped["status"].startswith("stopped: look_retention")
    assert dropped["look_retention"]["fired"] is True


def test_the_committed_report_records_the_five_bars(ident, confirm, calibration):
    block = calibration["confirm"]
    assert block["salt"] == confirm.SALT
    assert block["status"] in ("confirmed",) or block["status"].startswith("stopped: ")
    assert calibration["floor_status"] == block["status"]
    for arm in ("freesound", "cv_ar", "cv_other", "mistakes", "look_retention"):
        assert arm in block["arms"]
    assert block["arms"]["freesound"]["clips"] == confirm.FREESOUND_SIZE
    assert block["arms"]["cv_ar"]["clips"] == confirm.CV_AR_SIZE
    assert block["arms"]["cv_other"]["clips"] == 500
    assert block["arms"]["mistakes"]["clips"] == 52
    assert set(block["per_language"]) == set(confirm.PRIMARY_LOCALES)
    assert block["arms"]["cv_other"]["passed"] == sum(
        entry["passed"] for entry in block["per_language"].values())
    for arm in ("freesound", "cv_ar", "cv_other"):
        catches = block["catches"][arm]
        arms = block["arms"][arm]
        assert set(catches) == {"length", "margin", "both"}
        assert catches["length"] + catches["margin"] - catches["both"] == (
            arms["clips"] - arms["passed"])
        assert len(block["passing"][arm]) == arms["passed"]
        for item in block["passing"][arm]:
            assert {"key", "decode", "score", "length"} <= set(item)


# --- the oracle: recorded, never a selection step ----------------------------------------------


def test_the_oracle_is_recorded_and_never_feeds_b_u(ident, calibration):
    oracle = calibration["oracle"]
    assert set(oracle) == {"b_u_oracle", "clips_m_changed", "assignments_changed_at_b_u",
                           "artifact"}
    artifact = oracle["artifact"]
    assert artifact["sha256"] == _sha256(ident.ORACLE_PATH)
    assert artifact["path"] == "tools/tadabur/tlog/ident/oracle_margin.json"
    assert calibration["margin_definition"]["status"] == ident.oracle_consequence(
        calibration["bars"]["b_u"], oracle)
    assert calibration["margin_definition"]["status"] in ("final", "owner")
    # The bars are derived from the pinned margins only: no oracle argument, no oracle read.
    import inspect

    assert set(inspect.signature(ident.derive_bars).parameters) == {"look", "mistake", "grid"}

    assert ident.oracle_consequence(0.01, {"b_u_oracle": 0.01,
                                           "assignments_changed_at_b_u": 0}) == "final"
    assert ident.oracle_consequence(0.01, {"b_u_oracle": 0.02,
                                           "assignments_changed_at_b_u": 0}) == "owner"
    assert ident.oracle_consequence(0.01, {"b_u_oracle": 0.01,
                                           "assignments_changed_at_b_u": 3}) == "owner"


# --- decode stores, the commit order and the clean-tree refusal --------------------------------


def test_the_confirm_decode_stores_cite_the_doc_and_their_manifest(ident, confirm, calibration):
    manifests = {"freesound": confirm.FREESOUND_MANIFEST_PATH,
                 "cv_ar": confirm.CV_MANIFEST_PATH,
                 "cv_other": confirm.CV_MANIFEST_PATH}
    recorded_keys = {"freesound": "confirm_freesound", "cv_ar": "confirm_cv_ar",
                     "cv_other": "confirm_cv_other"}
    for arm, manifest in manifests.items():
        path = confirm.CONFIRM_DECODES[arm]
        store = json.loads(path.read_text(encoding="utf-8"))
        assert store["model"] == "base"
        assert store["identity"] == {"model_ref": ident.BASE_MODEL,
                                     "hub_revision": ident.BASE_REVISION}
        assert store["provenance"] == {
            "doc": {"path": "docs/tlog-ident-floor-preregistration.md",
                    "sha256": ident.FLOOR_PREREGISTRATION_SHA256},
            "manifest": {"path": str(manifest.relative_to(ROOT)), "sha256": _sha256(manifest)}}
        mode = "stream_b0" if arm == "freesound" else "whole-spans"
        assert set(store["fingerprints"]) == {mode}
        recorded = calibration["decodes"][recorded_keys[arm]]
        assert recorded == {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path),
                            "items": len(store["items"]), "mode": mode}


def test_the_commit_order_is_doc_then_manifests_then_decodes_then_scoring(ident, confirm):
    doc = _first_commit(ident.FLOOR_PREREGISTRATION_PATH)
    for manifest in (confirm.FREESOUND_MANIFEST_PATH, confirm.CV_MANIFEST_PATH):
        commit = _first_commit(manifest)
        assert commit != doc, f"{manifest.name} shares the doc's commit"
        assert _is_ancestor(doc, commit), f"{manifest.name} predates the doc"
    for store in confirm.CONFIRM_DECODES.values():
        commit = _first_commit(store)
        for manifest in (confirm.FREESOUND_MANIFEST_PATH, confirm.CV_MANIFEST_PATH):
            assert commit != _first_commit(manifest), f"{store.name} shares a manifest's commit"
            assert _is_ancestor(_first_commit(manifest), commit), f"{store.name} predates a manifest"
    scored = _last_commit(ident.CALIBRATION_PATH)
    for store in confirm.CONFIRM_DECODES.values():
        assert scored != _first_commit(store)
        assert _is_ancestor(_first_commit(store), scored)


def test_require_committed_passes_a_committed_file_and_refuses_any_drift(confirm, tmp_path):
    repo = tmp_path / "repo"
    (repo / "docs").mkdir(parents=True)
    doc = repo / "docs" / "doc.md"
    doc.write_text("frozen\n", encoding="utf-8")
    git = ["git", "-C", str(repo), "-c", "user.name=test", "-c", "user.email=test@example.com"]
    subprocess.run([*git, "init", "-q"], check=True)
    subprocess.run([*git, "add", "docs/doc.md"], check=True)
    subprocess.run([*git, "commit", "-qm", "doc"], check=True)

    assert confirm.require_committed([doc], repo=repo) is None

    doc.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="doc.md"):
        confirm.require_committed([doc], repo=repo)

    doc.write_text("frozen\n", encoding="utf-8")
    extra = repo / "docs" / "untracked.md"
    extra.write_text("scratch\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="untracked.md"):
        confirm.require_committed([extra], repo=repo)


def test_the_s5b_inputs_are_committed_before_the_decode_script_may_run(ident, confirm):
    confirm.require_committed([ident.FLOOR_PREREGISTRATION_PATH,
                               confirm.FREESOUND_MANIFEST_PATH, confirm.CV_MANIFEST_PATH])


@pytest.mark.slow
def test_the_confirm_bars_rederive_from_the_committed_decodes(ident, confirm, calibration):
    """The whole confirmed/stopped result, recomputed from the three committed stores, the 52
    mistake decodes and the look decodes. Marked slow: 2,000 fresh decodes are rescored."""
    database = ident.load_references()
    k = calibration["parameters"]["k"]

    def verdicts(path: Path) -> list[bool]:
        store = json.loads(path.read_text(encoding="utf-8"))
        return [ident.identify(item["spans"], database=database, k=k).floor_ok
                for item in store["items"].values()]

    freesound = verdicts(confirm.CONFIRM_DECODES["freesound"])
    cv_ar = verdicts(confirm.CONFIRM_DECODES["cv_ar"])
    cv_other = verdicts(confirm.CONFIRM_DECODES["cv_other"])

    clips = ident.real_mistake_clips()
    mistakes = ident.load_decode_store(ident.REAL_MISTAKE_DECODES_PATH)
    mistake_verdicts = [
        ident.identify(mistakes["items"][clip["audio_filename"]]["spans"],
                       database=database, filename_ayah=clip["surah_ayah"], k=k).floor_ok
        for clip in clips]
    look = [ident.identify(decode.spans, database=database,
                           filename_ayah=decode.surah_ayah, k=k)
            for decode in ident.load_look_decodes()]
    retention = sum(row.floor_ok for row in look) / len(look)

    bars = confirm.five_bars(freesound=freesound, cv_ar=cv_ar, cv_other=cv_other,
                             mistakes=mistake_verdicts, look_retention=retention)
    assert bars["status"] == calibration["floor_status"]
    assert bars["status"] == calibration["confirm"]["status"]
    for arm in ("freesound", "cv_ar", "cv_other", "mistakes", "look_retention"):
        reported = calibration["confirm"]["arms"][arm]
        assert reported["passed"] == bars[arm]["passed"], arm
        assert reported["fired"] == bars[arm]["fired"], arm
    assert calibration["look"]["retention"] == pytest.approx(retention)
    assert calibration["bars"]["provisional"] is (bars["status"] != "confirmed")

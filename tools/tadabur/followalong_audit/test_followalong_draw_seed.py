"""The draw's seed and shuffle idiom (#311, draw amendment 1 §1).

Committed before ``tadabur/followalong_audit/draw.py`` exists and **locked**: later turns may add
tests but never weaken or rewrite these. They pin ``docs/followalong-audit-draw-amendment.md``
§1 against the amendment alone, so any implementation is checkable without trusting the code.

Contract this file pins (``tadabur.followalong_audit.draw``)
-----------------------------------------------------------
* ``AMENDMENT_COMMIT``: M, the first ``main`` commit adding the amendment (PR #326).
* ``derive_seed(commit_hex) -> int``: the first 8 bytes, big-endian, of
  ``sha256("followalong-audit-draw:" + commit_hex)``, as an unsigned integer.
* ``subseed(seed, purpose) -> int``: the first 8 bytes, big-endian, of
  ``sha256(str(seed) + ":" + purpose)`` (``seed`` in decimal).
* ``seeded_order(items, seed, purpose, *, key=None) -> list``: ``items`` sorted ascending (by
  ``key`` when given), then shuffled once with ``random.Random(subseed(seed, purpose)).shuffle``.
  The input is not mutated and its order does not matter.
* ``draw_seed.json`` (``s5_draw/``): M, ``seed``, both documents' SHA-256 and D, the draw code's
  commit.
"""

from __future__ import annotations

import hashlib
import json
import random
import subprocess
from pathlib import Path

import pytest

from tadabur.followalong_audit.draw import (
    AMENDMENT_COMMIT,
    derive_seed,
    seeded_order,
    subseed,
)

ROOT = Path(__file__).resolve().parents[3]
DOCS = ROOT / "docs"
PREREG = DOCS / "followalong-audit-preregistration.md"
AMENDMENT = DOCS / "followalong-audit-draw-amendment.md"
DRAW_DIR = ROOT / "tools" / "tadabur" / "followalong_audit" / "s5_draw"

#: PR #326's squash commit: the first commit on main adding the amendment.
M = "03bc6d071a37a9bb264fcd6513381dbedee1c96e"
#: The formula's value for M, computed independently of the code under test.
SEED = 9665042067186208708
#: The frozen pre-registration's SHA-256 (the amendment's own header).
PREREG_SHA256 = "33d3ba26dbe6df3a72b0a03637934274a1c2526ac9e0dc306afd1545ea28ed39"


def _eight_bytes(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")


def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], check=True, capture_output=True,
                          text=True).stdout.strip()


def _main_ref() -> str:
    for ref in ("main", "origin/main"):
        if subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--verify", "-q", ref],
                          capture_output=True).returncode == 0:
            return ref
    pytest.skip("no main ref in this checkout")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- the formula --------------------------------------------------------------------------------
def test_the_amendment_commit_is_the_one_the_pull_request_landed():
    assert AMENDMENT_COMMIT == M


def test_the_seed_is_the_formulas_value_for_m_written_out_by_hand():
    assert _eight_bytes(f"followalong-audit-draw:{M}") == SEED
    assert derive_seed(M) == SEED


def test_the_seed_is_an_unsigned_64_bit_integer_that_changes_with_the_commit():
    other = "0" * 40
    assert 0 <= derive_seed(other) < 2**64
    assert derive_seed(other) == _eight_bytes(f"followalong-audit-draw:{other}")
    assert derive_seed(other) != derive_seed(M)


def test_a_subseed_hashes_the_seed_in_decimal_with_its_purpose():
    for purpose in ("order:pilot", "order:main", "repeats:pilot", "repeats:main", "ids",
                    "bootstrap", "cell:S1:tashkeel:yes"):
        assert subseed(SEED, purpose) == _eight_bytes(f"{SEED}:{purpose}")


def test_every_purpose_gets_its_own_subseed():
    purposes = ["order:pilot", "order:main", "repeats:pilot", "repeats:main", "ids", "bootstrap",
                "cell:S3:all", "cell:S4:all"]
    assert len({subseed(SEED, purpose) for purpose in purposes}) == len(purposes)


def test_a_subseed_known_value_pins_the_encoding():
    # sha256("9665042067186208708:order:pilot")[:8], big-endian
    assert subseed(SEED, "order:pilot") == 847086215277224432
    assert subseed(SEED, "ids") == 13477180233708760598
    assert subseed(SEED, "bootstrap") == 4323962964368679014


# --- the shuffle --------------------------------------------------------------------------------
def test_the_shuffle_is_the_sorted_items_through_python_random_on_the_subseed():
    items = [("b#0", 3), ("a#0", 9), ("a#0", 2), ("c#0", 1), ("b#0", 0), ("a#0", 5)]
    expected = sorted(items)
    random.Random(subseed(SEED, "cell:S3:all")).shuffle(expected)
    assert seeded_order(items, SEED, "cell:S3:all") == expected


def test_the_shuffle_does_not_depend_on_the_order_the_items_arrive_in():
    items = [f"item-{n:03d}" for n in range(40)]
    assert (seeded_order(items, SEED, "order:main")
            == seeded_order(list(reversed(items)), SEED, "order:main"))


def test_the_shuffle_leaves_the_input_alone_and_keeps_every_item():
    items = [3, 1, 2]
    assert sorted(seeded_order(items, SEED, "order:pilot")) == [1, 2, 3]
    assert items == [3, 1, 2]


def test_the_shuffle_sorts_by_the_key_when_one_is_given():
    rows = [{"key": "b", "position": 1}, {"key": "a", "position": 2}, {"key": "a", "position": 1}]
    expected = sorted(rows, key=lambda row: (row["key"], row["position"]))
    random.Random(subseed(SEED, "p")).shuffle(expected)
    assert seeded_order(rows, SEED, "p", key=lambda row: (row["key"], row["position"])) == expected


def test_another_seed_or_purpose_gives_another_order():
    items = list(range(60))
    base = seeded_order(items, SEED, "order:pilot")
    assert seeded_order(items, SEED + 1, "order:pilot") != base
    assert seeded_order(items, SEED, "order:main") != base


# --- the committed record -----------------------------------------------------------------------
@pytest.fixture(scope="module")
def draw_seed() -> dict:
    return json.loads((DRAW_DIR / "draw_seed.json").read_text(encoding="utf-8"))


def test_m_is_on_main_and_is_the_first_commit_adding_the_amendment():
    ref = _main_ref()
    first = _git("log", "--reverse", "--diff-filter=A", "--format=%H", ref, "--",
                 "docs/followalong-audit-draw-amendment.md").splitlines()[0]
    assert first == M
    subprocess.run(["git", "-C", str(ROOT), "merge-base", "--is-ancestor", M, ref], check=True)


def test_the_record_names_m_and_the_formulas_seed(draw_seed):
    assert draw_seed["amendment_commit"] == M
    assert draw_seed["seed"] == SEED == derive_seed(draw_seed["amendment_commit"])


def test_the_record_pins_both_documents_by_sha256(draw_seed):
    assert draw_seed["preregistration_sha256"] == PREREG_SHA256 == _sha256(PREREG)
    assert draw_seed["amendment_sha256"] == _sha256(AMENDMENT)


def test_the_record_names_the_draw_codes_commit(draw_seed):
    commit = draw_seed["draw_commit"]
    assert len(commit) == 40 and set(commit) <= set("0123456789abcdef")

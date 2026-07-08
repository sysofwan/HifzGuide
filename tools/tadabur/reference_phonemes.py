"""Reference phonemes for the Tadabur filter, cached by ``surah:ayah``.

Produces the normalized Hafs reference phoneme string for every ayah, reusing
``generate_phonemes.generate_reference_phonemes`` (quran-transcript + the 8-ayah
fallback) and applying the scorer's normalization (``tadabur.normalization``) so
the reference matches what the ``.balanced`` scorer compares against. Results are
cached to a single JSON file keyed by ``surah:ayah`` — computed once and reused
across all reciters. The cache is wrapped in a versioned envelope
(``CACHE_VERSION``, tied to ``normalization.ALGORITHM_VERSION``); a warm cache is
trusted only when it validates (matching version, complete 6236-ayah set, and
the normalized fallback sentinels), so a stale or partial cache is rebuilt rather
than silently reused. Regeneration is deterministic and idempotent.

Usage:
  python3 -m tadabur.reference_phonemes [--cache PATH] [--rebuild]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent.parent
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))

import generate_phonemes  # noqa: E402  (tools/ sibling module)

from . import normalization
from .normalization import normalize_phonemes

DEFAULT_CACHE_PATH = Path(__file__).parent / "cache" / "reference_phonemes.json"

# Cache-envelope schema/version. A warm cache is only trusted when it carries a
# matching version; this invalidates caches produced by an older normalization
# algorithm (``normalization.ALGORITHM_VERSION``) or an earlier envelope schema.
# Bump the schema part ("v1") if the on-disk envelope layout changes.
CACHE_VERSION = f"v1+norm{normalization.ALGORITHM_VERSION}"


def build_reference_phonemes(
    raw_phonemes: dict[str, str] | None = None,
) -> dict[str, str]:
    """Normalize every ayah's reference phoneme string, keyed by ``surah:ayah``.

    ``raw_phonemes`` defaults to the full quran-transcript generation (with the
    8-ayah fallback); pass an explicit mapping to normalize a specific set.
    """
    if raw_phonemes is None:
        raw_phonemes = generate_phonemes.generate_reference_phonemes()
    return {
        key: normalize_phonemes(phonemes).normalized
        for key, phonemes in raw_phonemes.items()
    }


def expected_fallback_sentinels() -> dict[str, str]:
    """Normalized reference for the 8 fallback ayat — the cheap content spot-check.

    These are stable, algorithm-derived values (no quran-transcript needed), so a
    warm cache carrying different fallback values was produced by a different
    normalization and must be rebuilt.
    """
    return build_reference_phonemes(dict(generate_phonemes.FALLBACK_PHONEMES))


def _is_valid_cache(payload: object) -> bool:
    """Whether an on-disk cache payload is trustworthy for the current algorithm.

    Requires the versioned envelope, the complete 6236-ayah key set, and the
    normalized fallback sentinels. Anything else (old bare-dict format, stale
    version, partial/truncated set, tampered fallbacks) is rejected so the cache
    is rebuilt rather than silently reused.
    """
    if not isinstance(payload, dict):
        return False
    if payload.get("cache_version") != CACHE_VERSION:
        return False
    references = payload.get("references")
    if not isinstance(references, dict):
        return False
    if len(references) != generate_phonemes.TOTAL_AYAT:
        return False
    return all(
        references.get(key) == value
        for key, value in expected_fallback_sentinels().items()
    )


def _read_valid_cache(cache_path: Path) -> dict[str, str] | None:
    """Return the cached references if a valid envelope is on disk, else ``None``."""
    if not cache_path.exists():
        return None
    try:
        with open(cache_path, encoding="utf-8") as f:
            payload = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None
    if not _is_valid_cache(payload):
        return None
    return payload["references"]


def _write_cache(cache_path: Path, references: dict[str, str]) -> None:
    """Write ``references`` in the versioned envelope, deterministically."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"cache_version": CACHE_VERSION, "references": references}
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2, sort_keys=True)


def load_reference_phonemes(
    cache_path: Path = DEFAULT_CACHE_PATH,
    rebuild: bool = False,
) -> dict[str, str]:
    """Return the cached reference phonemes, building and caching them on a miss.

    A warm cache is trusted only when it validates against the current algorithm
    (``_is_valid_cache``): a stale-version, partial, or corrupted cache is treated
    as a miss and rebuilt, so callers never receive references produced by an
    outdated normalization.
    """
    if not rebuild:
        cached = _read_valid_cache(cache_path)
        if cached is not None:
            return cached

    references = build_reference_phonemes()
    _write_cache(cache_path, references)
    return references


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Cache file path (default: {DEFAULT_CACHE_PATH})",
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help="Rebuild the cache even if it already exists.",
    )
    args = parser.parse_args()

    hit = not args.rebuild and _read_valid_cache(args.cache) is not None
    references = load_reference_phonemes(args.cache, rebuild=args.rebuild)
    print(
        f"{'Loaded' if hit else 'Built'} {len(references)} reference phonemes "
        f"({'cache hit' if hit else 'wrote'} {args.cache})"
    )


if __name__ == "__main__":
    main()

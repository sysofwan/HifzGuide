"""Reference phonemes for the Tadabur filter, cached by ``surah:ayah``.

Produces the normalized Hafs reference phoneme string for every ayah, reusing
``generate_phonemes.generate_reference_phonemes`` (quran-transcript + the 8-ayah
fallback) and applying the scorer's normalization (``tadabur.normalization``) so
the reference matches what the ``.balanced`` scorer compares against. Results are
cached to a single JSON file keyed by ``surah:ayah`` — computed once and reused
across all reciters. Regeneration is deterministic and idempotent; a warm cache
is a plain read.

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

from .normalization import normalize_phonemes

DEFAULT_CACHE_PATH = Path(__file__).parent / "cache" / "reference_phonemes.json"


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


def load_reference_phonemes(
    cache_path: Path = DEFAULT_CACHE_PATH,
    rebuild: bool = False,
) -> dict[str, str]:
    """Return the cached reference phonemes, building and caching them on a miss.

    A warm cache (``rebuild=False`` and the file present) is read directly, so
    the reference is computed once and reused across reciters/runs.
    """
    if cache_path.exists() and not rebuild:
        with open(cache_path, encoding="utf-8") as f:
            return json.load(f)

    references = build_reference_phonemes()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(references, f, ensure_ascii=False, indent=2, sort_keys=True)
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

    hit = args.cache.exists() and not args.rebuild
    references = load_reference_phonemes(args.cache, rebuild=args.rebuild)
    print(
        f"{'Loaded' if hit else 'Built'} {len(references)} reference phonemes "
        f"({'cache hit' if hit else 'wrote'} {args.cache})"
    )


if __name__ == "__main__":
    main()

"""How big the QUD correct-recitation eval set would be, before any audio is fetched (#144).

#144 writes a truth site at every target-pair carrier (D9: nine pairs, eighteen directions)
and every in-word geminate of the selected QUD mushafs (:mod:`tadabur.qud_selection`). A site
gets ``heard`` = the mushaf's letter only where the base teacher **and** ``h448`` both commit
it (ADR-0011 §3); anywhere else it waits for the owner's ear. This module estimates, per
selected mushaf and in total:

* **carriers**: from the Quran text of the ayat each mushaf covers (``coverage.json``), each
  ayah phonetized whole with its last word in waqf (:func:`tadabur.waqf_segments.
  hafs_segment_reference`), the eligibility rules of the Ship A census
  (:func:`tadabur.listening_session.pair_carriers`, ``held_geminates`` less those that cross
  a word boundary). A carrier of a letter in two pairs counts in both directions but is one
  site to listen to. Interior pauses, re-reads and the basmala QUD's text may carry are
  ignored; they move a few word-final sites only;
* **outcomes per carrier**, at the rates the two frozen decodes of the ``h448``-unseen
  Tadabur mining pool show (:func:`tadabur.power_frame.load_scored_pool`, read against the
  mushaf): *agreed* (both commit the mushaf's letter, or both hold the geminate), *disagreed*
  (either commits another letter, or one consonant at the geminate), *abstained* (neither
  disagrees, but one commits nothing at the carrier). ADR-0011 §3 lets only *agreed* take the
  mushaf; *disagreed* is what #144 sends to the owner; *abstained* is neither, and is
  counted apart for the owner to decide. The pool is crowd recitation, so these rates are an
  upper estimate for professional reciters;
* **listening minutes**: the seconds per site #61's sittings were planned with
  (``ship_a/worklist.summary.json``: its pair-partner and geminate parts, minutes per row);
* **disk**: the parquet download, and the same audio decoded to 16 kHz mono PCM_16.

Usage (from ``tools/``; torch-free, needs ``quran-transcript``; ~5 min, mostly the pool)::

  python -m tadabur.qud_sizing
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path

from training.tashkeel_eval import write_text_atomically

from .audio import TARGET_SAMPLE_RATE
from .listening_session import crosses_words, held_geminates, pair_carriers
from .qud_selection import COVERAGE_PATH, QUD_DIR, SELECTED, SELECTION_PATH
from .truth_sites import HELD, SHADDAH, TARGET_PAIRS

SIZE_PATH = QUD_DIR / "size_estimate.json"
WORKLIST_SUMMARY_PATH = Path(__file__).parent / "ship_a" / "worklist.summary.json"

GEMINATE = "shaddah:held:in_word"
AGREED, DISAGREED, ABSTAINED = "agreed", "disagreed", "abstained"
OUTCOMES = (AGREED, DISAGREED, ABSTAINED)
#: Bytes per second of 16 kHz mono PCM_16, the staged-audio format.
PCM16_BYTES_PER_SECOND = TARGET_SAMPLE_RATE * 2


def direction(prescribed: str, partner: str) -> str:
    return f"{prescribed}→{partner}"


def carrier_counts(reference: str, offsets: list[int]) -> Counter:
    """One realized reference's carriers: per pair direction, per carrier letter (each
    carrier once, ``letter:<c>``) and the in-word geminates (:data:`GEMINATE`)."""
    counts: Counter = Counter()
    letters = set()
    for carrier in pair_carriers(reference):
        counts[direction(carrier.prescribed, carrier.partner)] += 1
        letters.add((carrier.prescribed, carrier.index))
    counts.update(f"letter:{letter}" for letter, _ in letters)
    counts[GEMINATE] += sum(not crosses_words(reference, offsets, i)
                            for i in held_geminates(reference))
    return counts


def quran_carriers(
    reference_of: Callable[[list[str]], tuple[str, list[int]]],
    ayat: Iterable[tuple[int, int, list[str]]],
) -> tuple[dict[str, Counter], list[str]]:
    """:func:`carrier_counts` per ``"surah:ayah"`` over ``(surah, ayah, uthmani words)``,
    and the ayat the phonetizer cannot handle (left out, listed)."""
    counts, unsupported = {}, []
    for surah, ayah, words in ayat:
        key = f"{surah}:{ayah}"
        try:
            reference, offsets = reference_of(words)
        except (KeyError, IndexError):
            unsupported.append(key)
            continue
        counts[key] = carrier_counts(reference, offsets)
    return counts, unsupported


def _all_ayat() -> Iterable[tuple[int, int, list[str]]]:
    from quran_transcript import Aya

    for surah in range(1, 115):
        for ayah in range(1, Aya(surah, 1).get().num_ayat_in_sura + 1):
            yield surah, ayah, list(Aya(surah, ayah).get().uthmani_words)


def covered_ayat(surahs: Mapping[str, Mapping]) -> list[str]:
    """The ``"surah:ayah"`` keys a coverage summary's ``"1-7,9"`` ranges name."""
    keys = []
    for surah, entry in surahs.items():
        for part in entry["ayat"].split(","):
            low, _, high = part.partition("-")
            keys += [f"{surah}:{a}" for a in range(int(low), int(high or low) + 1)]
    return keys


# --- the pool's outcome rates ----------------------------------------------------------


def outcome(prescribed: str, h448: str | None, base: str | None) -> str:
    """One site's class from what each decode committed (``None``: nothing)."""
    if h448 == prescribed and base == prescribed:
        return AGREED
    if any(c is not None and c != prescribed for c in (h448, base)):
        return DISAGREED
    return ABSTAINED


def pool_rates(scored) -> dict[str, dict]:
    """Per carrier letter (``letter:<c>``, each pool carrier once) and for in-word geminates:
    the pool's sites and the share of each :data:`OUTCOMES` class."""
    seen: set = set()
    tallies: dict[str, Counter] = defaultdict(Counter)
    for one in scored:
        site = one.site
        if site.mark in TARGET_PAIRS:
            key = (site.audio_filename, site.start_sample, site.reference_index)
            if key in seen:
                continue
            seen.add(key)
            stratum = f"letter:{site.prescribed}"
        elif one.stratum == GEMINATE and site.mark == SHADDAH and site.prescribed == HELD:
            stratum = GEMINATE
        else:
            continue
        tallies[stratum][outcome(site.prescribed, one.h448.committed, one.base.committed)] += 1
    return {
        stratum: {"sites": sum(t.values()),
                  **{o: round(t[o] / sum(t.values()), 4) for o in OUTCOMES}}
        for stratum, t in sorted(tallies.items())
    }


def seconds_per_site(summary: Mapping) -> dict[str, float]:
    """#61's planned listening seconds per row: pair sites from its pair-partner part,
    geminates from its geminate part."""
    def per_row(part: str) -> float:
        entry = summary["parts"][part]
        return round(entry["minutes"] * 60 / entry["rows"], 2)

    return {"pair": per_row("pair_partner"), GEMINATE: per_row("geminate")}


# --- the estimate ----------------------------------------------------------------------


def _expected(carriers: int, rates: Mapping) -> dict:
    return {"carriers": carriers, **{o: round(carriers * rates[o], 1) for o in OUTCOMES}}


def estimate_mushaf(counts: Counter, rates: Mapping[str, Mapping],
                    seconds: Mapping[str, float]) -> dict:
    """One mushaf's (or a set's summed) carriers with expected outcomes and minutes."""
    directions = sorted(k for k in counts if "→" in k)
    letters = sorted(k for k in counts if k.startswith("letter:"))
    per_letter = {k: _expected(counts[k], rates[k]) for k in letters}
    geminates = _expected(counts[GEMINATE], rates[GEMINATE])

    def minutes(cls: str) -> float:
        pair = sum(e[cls] for e in per_letter.values()) * seconds["pair"]
        return round((pair + geminates[cls] * seconds[GEMINATE]) / 60, 1)

    return {
        "directions": {d: _expected(counts[d], rates[f"letter:{d[0]}"]) for d in directions},
        "letters": per_letter,
        "geminates_in_word": geminates,
        "listening_minutes": {DISAGREED: minutes(DISAGREED), ABSTAINED: minutes(ABSTAINED)},
    }


def build(selection: Mapping, coverage: Mapping, ayah_counts: Mapping[str, Counter],
          unsupported: list[str], rates: Mapping, seconds: Mapping, pool: Mapping) -> dict:
    """The estimate per selected mushaf and for the whole selection."""
    mushafs, total = {}, Counter()
    total_hours = total_bytes = 0.0
    for mushaf in selection["mushafs"]:
        if mushaf["decision"] != SELECTED:
            continue
        cov = coverage["mushafs"][mushaf["slug"]]
        counts: Counter = Counter()
        keys = covered_ayat(cov["surahs"])
        for key in keys:
            counts.update(ayah_counts.get(key, Counter()))
        hours = cov["duration_ms"] / 3_600_000
        mushafs[mushaf["slug"]] = {
            "reciter_id": mushaf["reciter_id"],
            "recording_year": mushaf["recording_year"],
            "ayat": len(keys),
            "ayat_unsupported": len(set(keys) & set(unsupported)),
            "hours": round(hours, 2),
            "download_bytes": mushaf["download_bytes"],
            "pcm16_bytes": round(hours * 3600 * PCM16_BYTES_PER_SECOND),
            **estimate_mushaf(counts, rates, seconds),
        }
        total += counts
        total_hours += hours
        total_bytes += mushaf["download_bytes"]
    return {
        "inputs": {"selection_qud_revision": selection["qud_revision"],
                   "coverage_qud_revision": coverage["qud_revision"],
                   "pool": pool},
        "rates": rates,
        "seconds_per_site": seconds,
        "ayat_unsupported_by_phonetizer": unsupported,
        "mushafs": mushafs,
        "total": {
            "mushafs": len(mushafs),
            "hours": round(total_hours, 2),
            "download_bytes": round(total_bytes),
            "pcm16_bytes": round(total_hours * 3600 * PCM16_BYTES_PER_SECOND),
            **estimate_mushaf(total, rates, seconds),
        },
    }


def main() -> None:
    from .power_frame import load_scored_pool
    from .waqf_segments import hafs_segment_reference

    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    coverage = json.loads(COVERAGE_PATH.read_text(encoding="utf-8"))
    summary = json.loads(WORKLIST_SUMMARY_PATH.read_text(encoding="utf-8"))
    ayah_counts, unsupported = quran_carriers(hafs_segment_reference(), _all_ayat())
    provenance, scored = load_scored_pool()
    estimate = build(selection, coverage, ayah_counts, unsupported, pool_rates(scored),
                     seconds_per_site(summary), provenance)
    write_text_atomically(SIZE_PATH, json.dumps(estimate, ensure_ascii=False, indent=1,
                                                sort_keys=True) + "\n")
    t = estimate["total"]
    print(f"{t['mushafs']} mushafs, {t['hours']} h, minutes {t['listening_minutes']}")


if __name__ == "__main__":
    main()

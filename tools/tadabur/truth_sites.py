"""Truth sites: one human-labelled position per record, independent of any model (#79).

ADR-0011 judges every model change against **truth** — what the reciter actually said —
never against the teacher or the mushaf. A truth site is the unit of that truth: one
labelled position in one piece of Tadabur audio, carrying

* **audio provenance** — Tadabur ``audio_filename``, ``shard``, the ``[start_sample,
  end_sample)`` span of the item, and ``audio_sha256`` of the staged 16 kHz WAV, so the
  audio can be re-downloaded and proven to be the audio that was labelled;
* **where** — ``surah_ayah``, the item's ``reference`` (the realized reference, the
  phonetizer's output for what was recited, waqf/wasl as recited) and ``reference_index``,
  the **carrier letter** in it that the mark under test sits on;
* **what** — the ``mark`` under test (a haraka, sukun, shaddah, or one soft pair), what the
  mushaf ``prescribed`` there and what the human ``heard`` (``unclear`` leaves the
  denominator);
* **how it was labelled** — the ``source``, whether the label ``assumes_competent_reciter``
  (the human adjudicated something else and the mark follows only if the reciter recited
  correctly), and the ``stratum`` with that stratum's ``stratum_population`` so an
  estimate over sampled sites can be weighted back to the population.

Provenance is filled in two steps. ``audio_filename`` and ``start_sample`` are known as
soon as a site is defined; ``shard``, ``end_sample`` and ``audio_sha256`` only once the
audio has been re-staged (#83). Those three are :data:`STAGING_FIELDS`: ``null`` together
(not yet staged) or set together (staged), never a mix, and never invented.

The on-disk form is JSONL, one :class:`TruthSite` per line, written by
:func:`write_truth_sites` and read back by :func:`load_truth_sites`, which validates every
row and the file as a whole. The labelled files live in ``truth_sites/`` next to this
module; that directory's ``README.md`` is the human-readable schema.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass, fields
from pathlib import Path

from training.tashkeel_worklist import VOWEL_NAMES

from .phoneme_sifat import soft_pair_contrasts
from .phoneme_vocab import PHONEME_ID_TO_CHAR

TRUTH_SITES_DIR = Path(__file__).parent / "truth_sites"

# --- where the label came from ---------------------------------------------------------
P35_FIXTURE = "p35_fixture"
WAQF_BOUNDARY = "waqf_boundary"
NEW_AUDIT = "new_audit"
SYNTHETIC_EDIT = "synthetic_edit"
SOURCES = frozenset({P35_FIXTURE, WAQF_BOUNDARY, NEW_AUDIT, SYNTHETIC_EDIT})

# --- what is under test, and the states a label may name -------------------------------
#: Haraka name -> its character in the realized reference.
HARAKA_CHARS: dict[str, str] = {name: char for char, name in VOWEL_NAMES.items()}
SUKUN = "sukun"
SHADDAH = "shaddah"
TASHKEEL_MARKS = frozenset(HARAKA_CHARS) | {SUKUN}
SOFT_PAIRS = soft_pair_contrasts()
MARKS = TASHKEEL_MARKS | {SHADDAH} | SOFT_PAIRS

HELD = "held"
NOT_HELD = "not_held"
GEMINATION_STATES = frozenset({HELD, NOT_HELD})
#: The listener could not tell. Kept in the file; the scorer leaves it out of denominators.
UNCLEAR = "unclear"

#: The plain consonants of the phoneme vocabulary (ids 1-28, ``ء`` .. ``ي``): the only
#: characters a mark can sit on. Madd (``ا ۥ ۦ``), ghunna and tashkeel are not carriers.
CONSONANTS = frozenset(PHONEME_ID_TO_CHAR[1:29])

# --- schema ----------------------------------------------------------------------------
#: Provenance known only once the audio is re-staged (#83): ``null`` together or set together.
STAGING_FIELDS = ("shard", "end_sample", "audio_sha256")

#: Tadabur's full config ships this many parquet shards (``tadabur.shard_reader``).
NUM_SHARDS = 385

_SHA256 = re.compile(r"[0-9a-f]{64}")
_SURAH_AYAH = re.compile(r"([1-9][0-9]*):([1-9][0-9]*)")


@dataclass(frozen=True)
class TruthSite:
    """One labelled position. See the module docstring and ``truth_sites/README.md``."""

    site_id: str
    source: str
    assumes_competent_reciter: bool
    audio_filename: str
    shard: int | None
    start_sample: int
    end_sample: int | None
    audio_sha256: str | None
    surah_ayah: str
    reference: str
    reference_index: int
    mark: str
    prescribed: str
    heard: str
    stratum: str
    stratum_population: int


SCHEMA_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(TruthSite))

# JSON type of every field, matched exactly: a bool never passes as an int.
_FIELD_TYPES: dict[str, type] = {
    "site_id": str,
    "source": str,
    "assumes_competent_reciter": bool,
    "audio_filename": str,
    "shard": int,
    "start_sample": int,
    "end_sample": int,
    "audio_sha256": str,
    "surah_ayah": str,
    "reference": str,
    "reference_index": int,
    "mark": str,
    "prescribed": str,
    "heard": str,
    "stratum": str,
    "stratum_population": int,
}
assert set(_FIELD_TYPES) == set(SCHEMA_FIELDS)


def label_states(mark: str) -> tuple[frozenset[str], frozenset[str]]:
    """The values ``prescribed`` and ``heard`` may take for ``mark``.

    A tashkeel mark is the mushaf's own mark, so it prescribes exactly itself, and the
    listener may hear any tashkeel mark. Shaddah is a gemination state: the mushaf
    prescribes ``held`` (a geminate) or ``not_held`` (a single consonant, the site of an
    *added* shaddah), and the listener heard either. A soft pair prescribes one of its two
    letters and the listener heard one of them.
    """
    if mark in TASHKEEL_MARKS:
        return frozenset({mark}), TASHKEEL_MARKS | {UNCLEAR}
    if mark == SHADDAH:
        return GEMINATION_STATES, GEMINATION_STATES | {UNCLEAR}
    letters = frozenset(mark.split("↔"))
    return letters, letters | {UNCLEAR}


def _check_types(data: dict, where: str) -> None:
    missing = set(SCHEMA_FIELDS) - data.keys()
    unknown = data.keys() - set(SCHEMA_FIELDS)
    if missing or unknown:
        raise ValueError(
            f"{where}: does not match the truth-site schema "
            f"(missing: {sorted(missing)}, unknown: {sorted(unknown)})"
        )
    for name, expected in _FIELD_TYPES.items():
        value = data[name]
        if value is None and name in STAGING_FIELDS:
            continue
        if type(value) is not expected:  # exact: a bool is not an int, an int not a float
            raise ValueError(
                f"{where}: {name} must be {expected.__name__}, got {value!r}"
            )


def _check_provenance(site: TruthSite, where: str) -> None:
    if not site.audio_filename:
        raise ValueError(f"{where}: audio_filename is empty")
    staged = [getattr(site, name) is not None for name in STAGING_FIELDS]
    if any(staged) and not all(staged):
        raise ValueError(
            f"{where}: {', '.join(STAGING_FIELDS)} are filled together when the audio is "
            f"staged, but only some are set"
        )
    if site.start_sample < 0:
        raise ValueError(f"{where}: start_sample {site.start_sample} is negative")
    if site.end_sample is not None and site.end_sample <= site.start_sample:
        raise ValueError(
            f"{where}: end_sample {site.end_sample} is not after start_sample "
            f"{site.start_sample}"
        )
    if site.shard is not None and not 0 <= site.shard < NUM_SHARDS:
        raise ValueError(f"{where}: shard {site.shard} is outside [0, {NUM_SHARDS})")
    if site.audio_sha256 is not None and not _SHA256.fullmatch(site.audio_sha256):
        raise ValueError(f"{where}: audio_sha256 is not 64 lowercase hex characters")


def _check_label(site: TruthSite, where: str) -> None:
    if site.source not in SOURCES:
        raise ValueError(f"{where}: unknown source {site.source!r}")
    if site.mark not in MARKS:
        raise ValueError(f"{where}: unknown mark {site.mark!r}")
    prescribable, hearable = label_states(site.mark)
    if site.prescribed not in prescribable:
        raise ValueError(
            f"{where}: mark {site.mark!r} cannot prescribe {site.prescribed!r} "
            f"(expected one of {sorted(prescribable)})"
        )
    if site.heard not in hearable:
        raise ValueError(
            f"{where}: mark {site.mark!r} cannot be heard as {site.heard!r} "
            f"(expected one of {sorted(hearable)})"
        )
    if not _SURAH_AYAH.fullmatch(site.surah_ayah) or int(site.surah_ayah.split(":")[0]) > 114:
        raise ValueError(f"{where}: surah_ayah {site.surah_ayah!r} is not 'surah:ayah'")
    if not site.stratum:
        raise ValueError(f"{where}: stratum is empty")
    if site.stratum_population < 1:
        raise ValueError(f"{where}: stratum_population must be positive")


def _check_reference(site: TruthSite, where: str) -> None:
    """The reference index must point at a carrier that bears what the site claims."""
    reference, index = site.reference, site.reference_index
    if not 0 <= index < len(reference):
        raise ValueError(f"{where}: reference_index {index} is outside the reference")
    carrier = reference[index]
    following = reference[index + 1] if index + 1 < len(reference) else ""
    if site.mark in SOFT_PAIRS:
        holds = carrier == site.prescribed
    elif carrier not in CONSONANTS:
        holds = False
    elif site.mark == SHADDAH and site.prescribed == HELD:
        holds = following == carrier  # the first of the doubled consonant
    elif site.mark == SHADDAH:
        preceding = reference[index - 1] if index else ""
        holds = carrier not in (following, preceding)  # a single consonant
    elif site.mark == SUKUN:
        holds = following not in HARAKA_CHARS.values()
    else:
        holds = following == HARAKA_CHARS[site.mark]
    if not holds:
        raise ValueError(
            f"{where}: reference[{index}] = {carrier!r} followed by {following!r} does not "
            f"carry {site.mark!r}"
        )


def parse_site(data: dict, where: str) -> TruthSite:
    """One validated :class:`TruthSite` from its JSON object; ``where`` prefixes errors."""
    _check_types(data, where)
    site = TruthSite(**data)
    _check_provenance(site, where)
    _check_label(site, where)
    _check_reference(site, where)
    return site


def _check_file(sites: list[TruthSite], where: str) -> None:
    """Invariants that span rows: unique ids, one item description, honest populations."""
    duplicates = sorted(k for k, n in Counter(s.site_id for s in sites).items() if n > 1)
    if duplicates:
        raise ValueError(f"{where}: duplicate site_id(s) {duplicates[:5]}")

    # The shard and checksum describe the clip's file; the rest describe one item in it.
    clips: dict[str, tuple] = {}
    items: dict[tuple[str, int], tuple] = {}
    for site in sites:
        clip = (site.shard, site.audio_sha256)
        item = (site.end_sample, site.surah_ayah, site.reference)
        if clips.setdefault(site.audio_filename, clip) != clip:
            raise ValueError(
                f"{where}: sites on {site.audio_filename} disagree on its shard or checksum"
            )
        if items.setdefault((site.audio_filename, site.start_sample), item) != item:
            raise ValueError(
                f"{where}: sites on {site.audio_filename}@{site.start_sample} disagree on "
                f"the item's end_sample, surah_ayah or reference"
            )

    populations: dict[str, set[int]] = {}
    for site in sites:
        populations.setdefault(site.stratum, set()).add(site.stratum_population)
    sampled = Counter(site.stratum for site in sites)
    for stratum, values in sorted(populations.items()):
        if len(values) != 1:
            raise ValueError(f"{where}: stratum {stratum!r} has populations {sorted(values)}")
        (population,) = values
        if population < sampled[stratum]:
            raise ValueError(
                f"{where}: stratum {stratum!r} holds {sampled[stratum]} sites but claims a "
                f"population of {population}"
            )


def audio_sha256(path: Path) -> str:
    """SHA-256 of a staged WAV file's bytes — the value ``audio_sha256`` records."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_audio(sites: list[TruthSite], audio_dir: Path) -> None:
    """Fail unless every site's staged audio under ``audio_dir`` matches its checksum.

    A site without a checksum cannot be verified and fails too: supplying audio is a
    request to prove the labels belong to it, and an unproven site is exactly the risk
    the checksum exists to remove.
    """
    unstaged = sorted({s.audio_filename for s in sites if s.audio_sha256 is None})
    if unstaged:
        raise ValueError(
            f"{len(unstaged)} item(s) have no audio_sha256 to verify, e.g. {unstaged[0]}"
        )
    expected = {s.audio_filename: s.audio_sha256 for s in sites}
    for filename, checksum in sorted(expected.items()):
        actual = audio_sha256(audio_dir / filename)
        if actual != checksum:
            raise ValueError(
                f"{audio_dir / filename}: sha256 {actual} does not match the recorded "
                f"{checksum}"
            )


def load_truth_sites(path: Path, audio_dir: Path | None = None) -> list[TruthSite]:
    """Load and validate a truth-site file, in file order.

    Every row must match the schema exactly (field set and JSON types), name a known
    source and mark with legal ``prescribed`` / ``heard`` states, carry the provenance
    required now, and point ``reference_index`` at a carrier that bears the mark. Across
    rows, site ids are unique, sites on one clip agree on its shard and checksum, sites on
    one item agree on its end, ``surah_ayah`` and reference, and each stratum has one
    population no smaller than its site count. With
    ``audio_dir``, every item's staged WAV must also match its recorded checksum
    (:func:`verify_audio`). Blank and ``#``-prefixed lines are ignored; a missing file
    raises, because a committed label file that vanished is an error, not an empty set.
    """
    sites: list[TruthSite] = []
    with open(path, encoding="utf-8") as f:
        for lineno, raw in enumerate(f, 1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            where = f"{path}:{lineno}"
            try:
                data = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: not valid JSON ({exc})") from exc
            if not isinstance(data, dict):
                raise ValueError(f"{where}: a truth site is a JSON object")
            sites.append(parse_site(data, where))
    _check_file(sites, str(path))
    if audio_dir is not None:
        verify_audio(sites, audio_dir)
    return sites


def write_truth_sites(sites: list[TruthSite], path: Path) -> None:
    """Atomically (over)write a truth-site file, validating every site first.

    Each site is round-tripped through :func:`parse_site` and the whole list through the
    file checks *before* anything touches disk, so the file is never left partially
    rewritten or holding a row :func:`load_truth_sites` would reject. One JSON object per
    line, keys sorted, Arabic left readable.
    """
    for number, site in enumerate(sites, 1):
        parse_site(asdict(site), f"write_truth_sites[{number}]")
    _check_file(sites, "write_truth_sites")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        for site in sites:
            f.write(json.dumps(asdict(site), ensure_ascii=False, sort_keys=True) + "\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)

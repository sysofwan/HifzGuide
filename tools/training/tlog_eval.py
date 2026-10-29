"""The tlog tranches as readout items (#227): gating rows only, the filename ayah in full.

A tlog item is one clip of an S7 manifest row (:mod:`tadabur.tlog_tranches`) and the
**filename ayah's full text** as its reference — words 1…N, whatever the decode records as
``word_from``/``word_to``. #212's committed records carry S6's identified span there on the
rows it identified (515 of the look gating rows are partial); the readout ignores it, and
:func:`span_trimmed_items` reads it only as the reported sensitivity line. N comes from the
reference database (the ``canonical_words`` word source), never from the span, so a planted or
stale span cannot change what is graded.

Gating is the S7 manifests' own stratum, read off the frozen row and never recomputed
(:data:`tadabur.tlog_tranches.GATING`). The ``reported:*`` strata and ``gating ∧ id_floor_ok``
are their own item sets (:func:`reported_item_sets`) and never enter gating. The bootstrap
cluster is the **surah** (the holdout pre-registration's S2 fallback): a clip is one user's
recitation of one ayah, and the manifests name no reciter. Confirm.1 and confirm.2 are one
pooled set before resampling (:func:`confirm_views`), each tranche still reported alone.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

from tadabur.qud_text import ItemReference, Unmapped, item_reference
from tadabur.tlog_tranches import CONFIRM_ROLES, GATING
from training.aug_readout_gate import TLOG_CONFIRM
from training.nonpro_decode import item_key
from training.nonpro_eval import ClipItem

#: The prefix of the strata a tranche reports beside gating.
REPORTED_PREFIX = "reported:"
#: Gating's id-floor sensitivity set: reported only, never pooled into gating.
ID_FLOOR_OK = f"{GATING}.id_floor_ok"
#: The tranches pooled into one confirm set (owner 2026-10-28), in S7's claim order: confirm.2
#: is opened by confirm.1's committed claim, so it is never read alone. The round reads the
#: first two of S7's confirm roles and never claims the rest (#227's pre-registration).
CONFIRM_TRANCHES = CONFIRM_ROLES[:2]
POOLED_CONFIRM = TLOG_CONFIRM
#: Why a tlog row has no readout item: the row cannot be graded on its filename ayah.
EXCLUDED_REASON = ("no item: the row names no ayah, or the phonetizer cannot handle the ayah "
                   "the row names")


def _canonical_words(label: str) -> list[str]:
    """The ayah's canonical words from the reference database: the item's whole ayah."""
    from tadabur.waqf_segments import _uthmani_words

    return _uthmani_words(label)


def _whole_ayah(row: Mapping, line: Mapping, words: Sequence[str]) -> Sequence[int]:
    """The filename ayah in full, whatever span the decode records."""
    return (1, len(words))


def _identified_span(row: Mapping, line: Mapping, words: Sequence[str]) -> Sequence[int]:
    """S6's identified span as the decode records it, read only as the reported sensitivity."""
    return (line.get("word_from") or 1, line.get("word_to") or len(words))


def _items(rows: Sequence[Mapping], lines: Mapping[str, Mapping], *, reference_of,
           words_of: Callable[[str], list[str]],
           segment_of: Callable[[Mapping, Mapping, Sequence[str]], Sequence[int]],
           stratum: str = GATING) -> tuple[list[ClipItem], list[str]]:
    """One item per row of ``stratum``, in key order, over the words ``words_of`` gives its
    filename ayah and the span ``segment_of`` cuts from them. A row with no committed decode is
    refused (a dropped row would silently shrink the population); a clip the phonetizer cannot
    handle, or one whose row names no ayah, is listed and left out."""
    items, unsupported = [], []
    for row in sorted((row for row in rows if row["stratum"] == stratum),
                      key=lambda row: item_key(row["audio_path"])):
        key = item_key(row["audio_path"])
        line = lines.get(key)
        if line is None:
            raise ValueError(f"{key}: no decode for a {stratum} row of the tranche")
        if any(line.get(field) != row[field] for field in ("split", "shard_file", "row")):
            raise ValueError(f"{key}: the decode is not the row's "
                             f"{row['split']}/{row['shard_file']}/{row['row']}")
        label = row["filename_label"]
        if label is None:
            unsupported.append(key)
            continue
        surah, ayah = (int(part) for part in label.split(":"))
        try:
            words = words_of(label)
            reference: ItemReference = item_reference(segment_of(row, line, words), words,
                                                      reference_of)
        except (KeyError, IndexError, Unmapped):
            unsupported.append(key)
            continue
        items.append(ClipItem(key, "tlog", row["audio_path"], surah, ayah,
                              line["start_sample"], line["end_sample"], row["audio_sha256"],
                              reference, "", surah,
                              {"assigned_label": row["assigned_label"],
                               "id_floor_ok": row["id_floor_ok"]}))
    return items, unsupported


def gating_items(rows: Sequence[Mapping], lines: Mapping[str, Mapping], *, reference_of,
                 canonical_words: Callable[[str], list[str]] | None = None
                 ) -> tuple[list[ClipItem], list[str]]:
    """The gating rows' items, each graded on its filename ayah's full text, and the clips the
    phonetizer cannot handle. ``lines`` are the committed decode records by item key."""
    return _items(rows, lines, reference_of=reference_of,
                  words_of=canonical_words or _canonical_words, segment_of=_whole_ayah)


def span_trimmed_items(rows: Sequence[Mapping], lines: Mapping[str, Mapping], *, reference_of,
                       canonical_words: Callable[[str], list[str]] | None = None
                       ) -> tuple[list[ClipItem], list[str]]:
    """The same items with their reference cut to the ``word_from``/``word_to`` the decode
    records: the reported sensitivity line, never what gating grades."""
    return _items(rows, lines, reference_of=reference_of,
                  words_of=canonical_words or _canonical_words,
                  segment_of=_identified_span)


def reported_item_sets(rows: Sequence[Mapping], lines: Mapping[str, Mapping], *, reference_of,
                       canonical_words: Callable[[str], list[str]] | None = None
                       ) -> dict[str, list[ClipItem]]:
    """Each reported stratum's items, and gating's ``id_floor_ok`` sensitivity set: reported
    only, never pooled into gating (round 2's pre-registration, "reported, not gating")."""
    words_of = canonical_words or _canonical_words
    names = sorted({row["stratum"] for row in rows
                    if row["stratum"].startswith(REPORTED_PREFIX)})
    if any(row["stratum"] == GATING and row["id_floor_ok"] is True for row in rows):
        names.insert(0, ID_FLOOR_OK)
    sets = {}
    for name in names:
        if name == ID_FLOOR_OK:
            chosen = [row for row in rows
                      if row["stratum"] == GATING and row["id_floor_ok"] is True]
            sets[name] = _items(chosen, lines, reference_of=reference_of, words_of=words_of,
                                segment_of=_whole_ayah)[0]
        else:
            sets[name] = _items(rows, lines, reference_of=reference_of, words_of=words_of,
                                segment_of=_whole_ayah, stratum=name)[0]
    return sets


def excluded_record(excluded: Mapping[str, str | None]) -> dict:
    """The rows an item builder left out, by key, with their filename ayat and why: written into
    the freeze and the look artifact so a graded population smaller than its manifest names the
    difference instead of silently dropping it."""
    return {"keys": sorted(excluded), "count": len(excluded),
            "ayat": sorted({label for label in excluded.values() if label}),
            "reason": EXCLUDED_REASON}


def confirm_views(tranches: Mapping[str, list[ClipItem]]) -> dict[str, list[ClipItem]]:
    """The pooled confirm set beside each tranche's own view: confirm.1 and confirm.2
    concatenated into one item set before resampling, each still read alone.

    Only the two claimable tranches may be pooled, in claim order (confirm.2 needs confirm.1's
    committed claim), and a clip may sit in one tranche only."""
    unknown = sorted(set(tranches) - set(CONFIRM_TRANCHES))
    if unknown:
        raise ValueError(f"{unknown[0]} is not pooled into {POOLED_CONFIRM}: the round reads "
                         f"{list(CONFIRM_TRANCHES)}")
    if CONFIRM_TRANCHES[1] in tranches and CONFIRM_TRANCHES[0] not in tranches:
        raise ValueError(f"{CONFIRM_TRANCHES[1]} is opened by {CONFIRM_TRANCHES[0]}'s committed "
                         "claim: pool it with confirm.1, never alone")
    seen: dict[str, str] = {}
    for name, items in tranches.items():
        for item in items:
            if seen.setdefault(item.key, name) != name:
                raise ValueError(f"{item.key} is in both {seen[item.key]} and {name}: a clip is "
                                 "pooled once")
    pooled = sorted((item for items in tranches.values() for item in items),
                    key=lambda item: item.key)
    return {POOLED_CONFIRM: pooled, **tranches}

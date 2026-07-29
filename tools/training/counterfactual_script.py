"""Build a recording sheet for the tashkeel counterfactual test.

:mod:`training.minimal_pairs` established what this corpus *cannot* settle: every clip in it
is correct recitation, so nothing in it distinguishes a model that **hears** a harakah from
one that reconstructs it from the Quran's fixed text. A context-aware text-only baseline
scores 0.9734 on the ambiguous slice — above every checkpoint measured — so the observed
~0.98 vowel recall is fully explainable without any hearing at all.

The only thing that separates the two is audio where the spoken vowel and the canonical
vowel **disagree**. Then the two hypotheses make opposite predictions:

* a model that hears transcribes the vowel that was *said*,
* a model that reconstructs transcribes the vowel the *text* says should be there.

This module picks the words worth recording. Each item is one word in one held-out ayah,
chosen so that:

* it carries **exactly one** short vowel, so the substitution and the scoring are both
  unambiguous — there is no second vowel to confound which one the model got right;
* its **context prior is deterministic** (the training text always vowelizes this word in
  this context the same way, with enough observations to be sure), so a reconstructing
  model has every reason to emit the canonical vowel and following the audio is decisive;
* the ayah is in the **val** split, so it was not in the fine-tune's training material.

Each item is recorded **twice by the same reciter**: once as written (``control``) and once
with the target vowel replaced (``counterfactual``). The control take is what makes a
negative result interpretable — if the model cannot decode the word correctly even when it
is recited correctly, that item says nothing about hearing, and it is dropped rather than
scored as "failed to follow the audio".

Substitutions are balanced across all six directed vowel swaps, so a model that hears one
colour but is deaf to another cannot hide behind an average.

Usage::

    python -m training.counterfactual_script \
        --manifest tadabur/audit_run/seg_v21/manifest_raw.jsonl \
        --labels   tadabur/audit_run/seg_v21/windowed_labels_v3.jsonl \
        --items 50 \
        --out-sheet  counterfactual_sheet.csv \
        --out-manifest counterfactual_items.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from training.minimal_pairs import (
    SHORT_VOWELS,
    read_segments,
    reference_sites,
    split_clips,
    text_prior,
)

FATHA, DAMMA, KASRA = "\u064e", "\u064f", "\u0650"
VOWEL_NAMES = {FATHA: "fatha (a)", DAMMA: "damma (u)", KASRA: "kasra (i)"}

# Every ordered pair of distinct vowels: a model deaf to one colour must not be able to
# hide behind an average over the others.
SWAP_DIRECTIONS = [(a, b) for a in (FATHA, DAMMA, KASRA) for b in (FATHA, DAMMA, KASRA) if a != b]

# A short vowel immediately followed by one of these is a **madd**: the vowel is the onset
# of an elongation, not a free choice. Substituting it is not naturally producible (there is
# no such word as مُا for مَا), and ADR-0003 documents madd carriers as an alignment-artifact
# source, so these words are excluded rather than recorded.
MADD_LETTERS = frozenset("\u0627\u0648\u064a\u06e5\u06e6")


# How many times the training text must show this word-in-context before its vowelization
# counts as something a reconstructing model could confidently have memorized.
MIN_CONTEXT_OBSERVATIONS = 3


@dataclass(frozen=True)
class CounterfactualItem:
    """One word to recite twice — as written, then with its vowel changed."""

    item_id: str
    surah_ayah: str
    segment_text: str
    word_index: int
    target_word: str
    canonical_vowel: str
    spoken_vowel: str
    reference_phonemes: str
    audio_filename: str

    @property
    def spoken_word(self) -> str:
        """The target word with its single short vowel replaced by the one to speak."""
        return "".join(
            self.spoken_vowel if c in SHORT_VOWELS else c for c in self.target_word
        )


def candidate_items(segments: list[dict], val_clips, priors) -> list[CounterfactualItem]:
    """Every held-out word that can carry a clean, decisive vowel substitution."""
    items = []
    for row in segments:
        if row["clip_audio_filename"] not in val_clips:
            continue
        words = row["uthmani"].split()
        sites = reference_sites(row["raw_reference_phonemes"], row["raw_word_offsets"])
        n_words = len(row["raw_word_offsets"]) - 1
        if len(words) != n_words:
            # The Uthmani text and the phoneme offsets must agree word-for-word, or the
            # sheet would point a reciter at the wrong word.
            continue
        for index, (start, end, site) in enumerate(_indexed(sites, row)):
            if len(site.reference_vowels) != 1:
                continue
            # The Uthmani spelling must also carry exactly one short vowel. The phoneme
            # form and the written form do not always agree on vowel count, and
            # ``spoken_word`` rewrites the written form -- so without this the sheet would
            # tell a reciter to change two vowels when only one is under test.
            if sum(1 for c in words[index] if c in SHORT_VOWELS) != 1:
                continue
            phonemes = row["raw_reference_phonemes"][start:end]
            if _vowel_is_madd(phonemes):
                continue
            counts = priors.context.get(site.context_key)
            if not counts or len(counts) != 1:
                continue
            if sum(counts.values()) < MIN_CONTEXT_OBSERVATIONS:
                continue
            items.append(
                CounterfactualItem(
                    item_id="",
                    surah_ayah=row["surah_ayah"],
                    segment_text=row["uthmani"],
                    word_index=index,
                    target_word=words[index],
                    canonical_vowel=site.reference_vowels,
                    spoken_vowel="",
                    reference_phonemes=phonemes,
                    audio_filename=row["audio_filename"],
                )
            )
    return items


def _vowel_is_madd(phonemes: str) -> bool:
    """Whether the word's single short vowel opens an elongation."""
    return any(
        c in SHORT_VOWELS and i + 1 < len(phonemes) and phonemes[i + 1] in MADD_LETTERS
        for i, c in enumerate(phonemes)
    )


def _indexed(sites, row):
    """Sites paired with their word index, recovered from the offsets."""
    offsets = row["raw_word_offsets"]
    position = {start: i for i, start in enumerate(offsets[:-1])}
    return [(start, end, site) for start, end, site in sites if start in position]


def select(items: list[CounterfactualItem], count: int, seed: int = 0) -> list[CounterfactualItem]:
    """Choose a balanced, deterministic subset spanning all six swap directions.

    One ayah contributes at most one item, so a single mis-recited ayah cannot dominate the
    result, and the items stay closer to independent than a per-word sample would.
    """
    rng = random.Random(seed)
    by_direction: dict[tuple[str, str], list[CounterfactualItem]] = {d: [] for d in SWAP_DIRECTIONS}
    for item in items:
        for direction in SWAP_DIRECTIONS:
            if item.canonical_vowel == direction[0]:
                by_direction[direction].append(item)
    for bucket in by_direction.values():
        rng.shuffle(bucket)

    chosen: list[CounterfactualItem] = []
    used_ayahs: set[str] = set()
    # Distinct words as well as distinct ayahs: three recordings of بَلْ probe one lexical
    # item three times, which is far less informative than three different words.
    used_words: set[str] = set()
    cursors = {direction: 0 for direction in SWAP_DIRECTIONS}

    # Round-robin rather than filling each direction in turn. Damma is much rarer than
    # fatha, and words are deduplicated globally, so a greedy pass drains the shared pool
    # and starves whichever direction is drawn last -- silently leaving a vowel swap
    # completely untested.
    while len(chosen) < count:
        progressed = False
        for direction in SWAP_DIRECTIONS:
            if len(chosen) >= count:
                break
            bucket = by_direction[direction]
            while cursors[direction] < len(bucket):
                item = bucket[cursors[direction]]
                cursors[direction] += 1
                if item.surah_ayah in used_ayahs or item.target_word in used_words:
                    continue
                used_ayahs.add(item.surah_ayah)
                used_words.add(item.target_word)
                chosen.append(
                    CounterfactualItem(
                        **{
                            **asdict(item),
                            "item_id": f"cf{len(chosen):03d}",
                            "spoken_vowel": direction[1],
                        }
                    )
                )
                progressed = True
                break
        if not progressed:
            break
    return chosen


def write_sheet(items: list[CounterfactualItem], path: Path) -> None:
    """A human recording sheet: what to say, and what to say differently."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "item_id", "surah_ayah", "recite_this_phrase", "target_word",
            "word_position", "normally_says", "instead_say", "word_as_spoken",
            "take_1_file", "take_2_file",
        ])
        for item in items:
            writer.writerow([
                item.item_id,
                item.surah_ayah,
                item.segment_text,
                item.target_word,
                item.word_index + 1,
                VOWEL_NAMES[item.canonical_vowel],
                VOWEL_NAMES[item.spoken_vowel],
                item.spoken_word,
                f"{item.item_id}_control.wav",
                f"{item.item_id}_counterfactual.wav",
            ])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--items", type=int, default=50,
                        help="how many words to record (each is recorded twice).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-sheet", type=Path, required=True)
    parser.add_argument("--out-manifest", type=Path, required=True)
    args = parser.parse_args()

    segments = read_segments(args.manifest)
    train_clips, val_clips = split_clips(args.labels)
    priors = text_prior(segments, train_clips)
    candidates = candidate_items(segments, val_clips, priors)
    chosen = select(candidates, args.items, args.seed)
    if len(chosen) < args.items:
        print(f"warning: only {len(chosen)} of {args.items} items available", flush=True)

    write_sheet(chosen, args.out_sheet)
    args.out_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.out_manifest.open("w", encoding="utf-8") as handle:
        for item in chosen:
            handle.write(json.dumps(asdict(item), ensure_ascii=False) + "\n")

    by_direction: dict[str, int] = {}
    for item in chosen:
        key = f"{VOWEL_NAMES[item.canonical_vowel]} -> {VOWEL_NAMES[item.spoken_vowel]}"
        by_direction[key] = by_direction.get(key, 0) + 1
    print(f"{len(candidates)} candidate words; selected {len(chosen)} across "
          f"{len({i.surah_ayah for i in chosen})} ayahs.")
    for key, n in sorted(by_direction.items()):
        print(f"  {key}: {n}")
    print(f"Wrote {args.out_sheet} and {args.out_manifest}")
    print(f"Record {2 * len(chosen)} clips total (one control + one counterfactual per item).")


if __name__ == "__main__":
    main()

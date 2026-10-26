# tlog S5 ayah identification: pre-registered amendment (#190)

**Status:** frozen 2026-10-25, pre-outcome. No Tarteel v1 look clip had been scored when this was
written, including the 1,000-clip shortlist guard. This file amends the rules that
[`tlog-holdout-preregistration.md`](tlog-holdout-preregistration.md) freezes (SHA-256
`204fe7f980488aff73402c56f1729f7f17f7c9471ea0664f6365bb8870b62086`); that document is not edited.
It fills in procedures "Ayah identification" leaves open, and corrects one count. No threshold rule
changes.

**Disclosure.** Before this was written the scout measured the Tarteel v1 look decodes' length-bin
counts, the `cer_wav ≥ 0.2` tail count, and the wall-clock cost of edit-distance alignments. No
identification score, floor, Δ, margin or label was recorded on any look clip.

## Amendment to "Ayah identification"

- Amended 2026-10-25 (S5, pre-outcome; no Tarteel v1 look clip had been scored when this was
  written). These pins fill in procedures the text above leaves open, and correct one count. No
  threshold rule changes.
  - **Alphabet:**
    - decodes and references are compared after `normalize_phonemes` (algorithm version recorded);
    - each ayah is normalised separately, and word offsets are remapped through its `offset_map`;
    - `len(decode)` is the normalised length;
    - words are quran-transcript `uthmani.split()` words;
    - the 8 `FALLBACK_PHONEMES` ayat take boundaries 0…n−2 from phonetizing `words[:-1]`, and the
      last word's start from an infix alignment of that prefix into the fallback string;
    - text classes are equality of the normalised per-ayah references.
  - **Decoys:** each normalised ayah reference is reversed by grapheme cluster (a base plus its
    combining marks), and the reversed ayat are concatenated in ayah order per surah.
  - **Shortlist:**
    - consonant-skeleton 3-grams over windows of an ayah ± 1;
    - a window's rank is its number of distinct query 3-grams, ties by (surah, ayah) of the centre
      ayah;
    - shortlist recall for the guard means the shortlist's best score equals the best score over all
      windows;
    - the library is `edlib` 1.3.9.post1, mode `HW`.
  - **The best decoy score uses the same search as the real one:** the same shortlist construction
    over the decoy sequences, the same k, the same scoring. The best over all decoy windows is
    reported, never used.
  - **Real-mistake clips:** the bar on b uses the **52** clips that hold the 56 sites ("42" above was
    the reciter count). They are listed in
    `tools/tadabur/tlog/ident/real_mistake_clips.jsonl`
    (sha256 `6b8c4002b098a77218d68e3cfa730b088703b37dd03a9c3768df2ffbcbec59df`).

## Errata

- **Relabel rule.** "0 of the 42 clips holding the 56 owner-verified real-mistake sites" reads **0 of
  the 52 clips**; 42 was the reciter count. This makes the bar stricter, since more clips must not
  be relabelled. The 52 clips are the pinned `real_mistake_clips.jsonl` above.

# Follow-along audit: draw amendment 1 — how the frozen design is drawn (2026-11-25)

**Status:** dated 2026-11-25, owner-approved the same day **before any draw**, and frozen the day
it lands. This is an amendment, not an edit: the frozen
[`docs/followalong-audit-preregistration.md`](followalong-audit-preregistration.md) (SHA-256
`33d3ba26dbe6df3a72b0a03637934274a1c2526ac9e0dc306afd1545ea28ed39`) stays byte-identical, and the
draw pins the SHA-256 of both documents.

Issue: [#311](https://github.com/sysofwan/HifzGuide/issues/311). Parent: #305 (the follow-along
audit); the frozen design is #310's.

**What this changes.** It changes **no frozen number and no rule**: not the frame (49,000 words),
the sizes (540; pilot 150; S1 200 / S3 250 / S4 60 / S5 30; 25 repeats), the attribution
precedence, nor the decision rule's thresholds. It fills in the details the draw code needs and
the frozen text leaves open. Each item says whether it is a **clarification** (one reading of the
existing text) or an **addition** (something the text never set). Item 2 can lower a cell's N, but
only through the frozen document's own reproduction rule. Item 1's seed carries three precisions
that close a seed-shopping gap — the shuffle is pinned here, not in code written after the seed is
known — and they change no number either.

Owner approval: https://github.com/sysofwan/HifzGuide/issues/311#issuecomment-7168352666

Principal cross-check: https://github.com/sysofwan/HifzGuide/issues/311#issuecomment-7168475180

## 1. The random seed (addition)

- The frozen text says "seeded" but names no seed. The seed is derived from the commit that first
  adds this file to `main`, so it is unknown until this file merges and nobody can shop for it.
- **The commit.** M is the **first** commit on `main` whose tree adds
  `docs/followalong-audit-draw-amendment.md`: the first line of
  `git log --reverse --diff-filter=A --format=%H main -- docs/followalong-audit-draw-amendment.md`.
  A revert and re-merge therefore does not produce a second, chosen M.
- **The seed.** `seed` is the 64-bit unsigned integer from the first 8 bytes, big-endian, of
  `sha256("followalong-audit-draw:" + M_hex)`, with `M_hex` the 40-character lowercase
  hexadecimal SHA of M.
- **Sub-seeds.** Each use gets its own: the same 8 bytes of
  `sha256(str(seed) + ":" + purpose)`, `seed` in decimal — `cell:<stratum>:<cell>` for each cell's
  permutation, `order:pilot`, `order:main`, `repeats:pilot`, `repeats:main`, `ids` for the blind
  item ids' salt, and `bootstrap`.
- **The shuffle is pinned here, not in code written after the seed is known.** Each cell's words
  are listed in ascending `(key, position)` order and shuffled once with Python's
  `random.Random(subseed).shuffle`; the pilot is the front of that list and the main sitting
  continues it. The serve orders and the repeat positions use the same idiom on their own
  sub-seeds. Any implementation is therefore checkable against this document alone, whenever it
  lands.
- **`draw_seed.json`** records M, `seed`, both documents' SHA-256, and D, the commit of the draw
  code that produced the files. D must be an ancestor of M; **or**, where the approved split lands
  the draw after the amendment, the re-derivation test must reproduce every drawn file from a
  clean checkout at D.
- A test asserts that M is on `main`, that M is the first commit adding this file, and that
  `seed` is the formula's.

## 2. Words that do not reproduce (clarification; may lower N)

- §5 says S5 regenerates every App trace, and that a word whose regenerated grade does not match
  the committed table "is listed and never drawn".
- Such a word leaves the frame into a reported **"did not reproduce"** pool with π = 0, as the
  coverage gate's uncovered words do.
- Its cell's N, and so π = n ÷ N, counts only the words that reproduced. Each cell is shuffled in
  its frozen frame order, and a removed word is skipped when the prefix is taken, so each cell
  stays a simple random sample.
- If every word reproduces (expected, given 25/25 identical in S3's determinism check), every N
  stays as tabulated.
- The gate's run note reports the population-wide flip count against the committed `word_grades`
  table, and how many of those flips fall in S1 cells — the tabulated cells whose N a removal
  would move.

## 3. Pilot ties (clarification)

- §3: the pilot is each cell's prefix, "allocated by largest remainder over the cells' main
  allocations with no floor".
- Ties use the rule that produced the frozen sizes (`allocation.allocate`): the larger main
  allocation, then the alphabetical cell name. This reproduces the frozen 56 / 69 / 17 / 8 exactly.
- It gives **S5: interior 3, leading 3, trailing 2**.
- It gives **S1's 56**: other_consonant 12 / 17; tashkeel 3 / 5; soft_pair 1 / 4;
  skipped 1 / 3; unrecorded 1 / 4; shaddah 1 / 2; target_pair 1 / 1 (each pair is
  base-detector no / yes).

## 4. Repeats (clarification)

- Any served item can be repeated, whatever its stratum.
- **Pilot:** the 150 are put in a seeded random order. 7 originals are picked at random among the
  items served in positions 1–130, and each copy is placed at a random later position at least 20
  items after its original. The pilot is 157 served items.
- **Main sitting:** the same, with 18 repeats over the main items.
- A repeat looks exactly like an original. Only its first verdict counts in any rate, and
  agreement is reported on the five top-level verdicts.

## 5. Which misses count toward "60" (clarification)

- Rule 4's **60 confirmed misses** counts every listened word, in **any** stratum, where you
  answered **mistake** and the App did not flag it. First verdicts only, unweighted.
- This matches §5 (rule 2's share uses all listened words) and §8 (the resize adds to whichever
  stratum misses most, S4 and S5 included).
- The **headline miss rate** stays on H = S1 ∪ S3 ∪ S5-interior, as §7 says.
- The owner confirmed this reading with the amendment's approval; the narrower H-only reading was
  considered and is not taken.

## 6. "First and last word" (clarification)

Every clip is one ayah. The edge words in the §3/§7 sensitivity are that ayah's first and last
**Muraja phoneme words**: the same word units as the population, and the same edges as the S5
leading/trailing classes.

## 7. Rule 1's count (clarification)

- "≥ 3 instances in ≥ 3 clips" is counted **per bucket**: F1, F1-align, M1r, M1, M1u, and
  M-nograde when it falls on the follow-along side. It counts listened words with that first
  bucket, in at least three different clips, across all strata.
- F-ref is reported beside them as a reference suspect.
- The scorer computes the model share **before** any fix and labels it so. The share "after the
  follow-along fixes" is recomputed by a corpus replay once a fix exists (rule 2).

## 8. "Half of F3" (clarification)

It means **at least half** (≥ 50%), as #310's issue text says ("≥ half of F3") and as rule 3's
"≥ 50%" reads.

## 9. A mistake the App did flag (clarification)

This is a **correct catch**. It is not a false flag and not a miss. It enters the miss rate's
denominator (confirmed mistakes) and never its numerator. It is a verified mistake for the
dataset rows (§9). The mistake kinds, including "added", describe the mistake; they do not change
any rate.

## 10. Confidence intervals (addition)

Every 95% interval is a percentile bootstrap that resamples whole clips: 4,000 resamples (the
repo's convention, `cluster_bootstrap`), seeded from item 1. The full-frame sensitivity in §7
uses the same method.

## Changelog

- 2026-11-25: written, owner-approved and frozen before S5's draw. Ten resolutions to gaps the
  frozen text leaves open — two additions (the seed, the bootstrap) and eight clarifications —
  with the principal's three seed precisions folded into item 1 and the gate's flip report into
  item 2. No frozen number or rule moves.

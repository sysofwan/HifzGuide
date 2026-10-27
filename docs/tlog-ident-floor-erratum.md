# Erratum to the S5b floor pre-registration (#204): three disclosures, measured

**Date:** 2026-10-27.
**Status:** disclosure only. It changes no threshold, no draw, no decode store and no bar.

It amends `docs/tlog-ident-floor-preregistration.md` (SHA-256
`7bd2e1958ea8330df472030aefb6c97de7d0e9ccc2aa81ac133db682e1d58e3a`), which is frozen and is never
edited; the margin definition stays as pinned in `docs/tlog-ident-margin-amendment.md` (SHA-256
`eded44e95dc58a3d5ccb86659f296eb50b8b5e0cf236a21b0e5acb5258e3aba2`).

The confirmed run (`tools/tadabur/tlog/ident/ident_calibration.json`, `floor_status:
"confirmed"`) exposed three things a reader needs before trusting or reproducing the result:
(a) the pre-registration's disclosed post-look look figures are an older engine's, and one search
change separates them from the confirmed run's; (b) the Quran-text guard the draw used cannot
match Urdu or Persian orthography, and the fold that fixes it excludes none of the committed
draw; (c) two provenance facts that otherwise read as corruption.

## (a) The disclosed look figures are the S5 measurement's, one search change back

The pre-registration's §3 table discloses "Tarteel look kept 99.78%" and "look tail
(`cer_wav ≥ 0.2`) kept 99.72%". Those were measured in S5 on the engine architect r16 names,
`dc602df` — the commit whose look decodes the table was read from. That engine already includes
`79acded` (one normalisation per decode, not one per call) and `c851fa0` (span attribution to the
frozen rule): `git merge-base --is-ancestor 79acded dc602df` succeeds.

One score change since then moves the figures: `659bce0` (S5's self-check r1) ranks the decoy
shortlist with the decoy 3-gram index, where the measurement had ranked it with the real index.
Rebuilding the old search on the committed look decodes at the committed `k = 20` reproduces the
disclosed numbers exactly, and the decoy index gives the ones the confirmed run reports:

| set | real-index decoy search (`dc602df`) | decoy-index search (`659bce0` on, committed) |
|---|---|---|
| Tarteel look kept | 99.78% (4,527/4,537) | 99.63% (4,520/4,537) |
| look tail (`cer_wav ≥ 0.2`) kept | 99.72% (1,801/1,806) | 99.50% (1,797/1,806) |

Nine rows differ between the two searches. 17 look rows fail the rule on the committed engine: 3
have fewer than `L_min = 8` normalised characters (none empty), and the other 14 fail on the
margin.

The pre-registration was **not** written before this engine existed. At its own commit `1c3ddd9`,
`tools/tadabur/tlog_ident.py` is byte-identical to main's fixed engine (blob
`5f7eb4047e6c50306d4ef75732f4287a26e2b8be`); the document copied r16's stale disclosure table
into a document written after the engine that confirms it.

What does stand is the post-look disclosure: both values were **picked after the look half had
been scored**. Architect r16 chose `L_min = 8` over 10 *for look retention* and `δ = 0.30` over
0.20 after reading its neighbourhood table (every `L` 6–10 × `δ` 0.20–0.40 gives 0–3 of the 445
noise recordings passing and ≥ 99.4% look kept). No value was chosen on any confirmation clip;
the three fresh sets exist to test that choice. Look retention's bar is 90%, so the committed
99.63% is inside it, and `b` and `b_u` read Δ and the margin, never retention, so no bar moves.

`test_the_disclosed_look_figures_reproduce_with_the_real_index_decoy_search`
(`tools/tadabur/test_tlog_ident_confirm.py`, slow tier) re-derives both columns from the
committed look decodes.

## (b) The Quran-text guard does not match Urdu or Persian letter variants

The pre-registration's §4c applies the guard to `ur` and `fa` because those clips are written in
Arabic script. The guard normalises with `normalize_imlai`, which unifies alef, ta marbuta and
alef maqsura only; Urdu's heh goal (`ہ`) and yeh barree (`ے`) and Persian's farsi yeh (`ی`) stay
as they are, so a Quran quote in either orthography does not match:

```
guard("بِسْمِ اللَّهِ الرَّحْمَنِ الرَّحِيمِ")  -> True    (Arabic)
guard("بسم اللہ الرحمن الرحیم")                -> False   (Urdu, unfolded)
guard("بسم الله الرحمن الرحیم")                -> False   (Persian, unfolded)
```

`tools/tadabur/tlog/ident/confirm/common_voice.json` records `excluded_guard: 0` for both, which
is that mismatch's signature, not evidence of a clean corpus.

Re-checked with the fold (`ھ ہ ۂ → ه`, `ی ے → ي`) over all 1,000 committed sentences: **0
excluded**, including 0 of the 200 `ur`/`fa` clips
(`test_the_folded_guard_catches_urdu_and_persian_basmala_and_excludes_none_of_the_draw`). The
drawn sets are clean under either reading, so **the draw stands as committed**: it was not
redrawn, and the manifest keeps the sentences the unfolded guard admitted.

`tadabur.tlog_ident_confirm.fails_folded_quran_guard` is the folded check, for future draws and
for this re-check. No committed sentence, manifest or store was edited.

## (c) Two provenance facts

**`samples_sha256` is host-scoped.** The manifest's field hashes the decoded samples on the Mac
that drew the row; the store's hashes the samples the GPU box decoded from the same MP3. MP3
decoding and resampling are host- and library-dependent, so the two differ (0 of 1,000 equal)
without any corruption. The decode matches a drawn row by `freesound_id`, licence, username and
duration (±0.1 s), and the store records the hash of what it actually decoded; the store's keys
are the manifest's 1,000 ids. The Common Voice arm keeps a byte-level `audio_sha256` check, which
is host-independent.

**The provenance chain's commits are branch commits.** `confirm.provenance` cites `1c3ddd9` (the
doc), `cd46731` / `5a85d48` (the manifests), `27c7598` (the three stores) and `272783d` (the
scoring run). A squash merge collapses those into one commit on main, so they are recoverable
only through `refs/pull/205/head` (PR #205). The artifacts record a null `pull_request` pair
because they were produced before the PR opened; `tadabur.tlog_index.provenance()` is the
canonical helper for a future artifact's pair.

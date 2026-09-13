# Neighbour-ayah bleed: detecting it, and clipping it

Settles HifzGuide #67 / Muraja ADR-0016 decision 3. A Tadabur clip is cut from a continuous
recitation, so it often carries audio of the ayah before or after it — a **lead-in** or a
**trailing bleed** (ADR-0002). For a passing clip `waqf_detect.segment_clip` re-cuts that away
and records the span on `ClipStatus`. For a *rejected* clip it does not: `repeated_recitation`
is a skip reason, and a skipped clip's recitation span defaults to the whole clip. The reject
pile is exactly the population ADR-0016 mines, so bleed reached that corpus unfiltered.

Everything below is measured on the shard-20 reject pile (71 clips of 1,000) and on the nine
human-adjudicated low-band clips inside it. Reproduce with:

```bash
python -m tadabur.bleed_detect --rejects reject_run/rejects.jsonl --json bleed.json
python -m tadabur.bleed_recut --decodes bleed_run/decodes.jsonl \
  --clips bleed_run/clips/ --out bleed_run/recuts.jsonl        # needs the GPU
```

## Why the first three signals could not work

| signal | leading P/R | trailing P/R | clip P/R |
| --- | --- | --- | --- |
| `leading_trim` / `trailing_trim` >= 5 | 0.67 / 1.00 | 1.00 / 0.50 | **0.67 / 0.50** |
| per-ayah reference comparison (this / prev / next, quarter slices) | 1.00 / 0.00 | 1.00 / 0.25 | **1.00 / 0.25** |
| three-ayah anchored alignment (this work) | 1.00 / 1.00 | 1.00 / 1.00 | **1.00 / 1.00** |

`waqf_detect`'s edge re-cut is not a fourth row: `recut_start`/`recut_end` are
`decode_times[alignment.query_start]` and `decode_times[alignment.query_end - 1]`, the very span
the trims are computed from, so it scores identically to row 1 by construction.

The trims and the re-cut fail for one reason. Smith-Waterman is **local** with cheap affine gap
extension (`GAP_OPEN -0.5`, `GAP_EXTEND -0.1`), so a query running off the end of its reference
does not stop — it drags the alignment through the unmatched audio rather than paying to
restart. That is the drag Muraja ADR-0009 documents. A dragged bleed leaves **both trims at 0**,
which is precisely what 2:278 and 12:48/spk0002 look like: two-phoneme bleeds, `trim = 0` and
`trim = 2`, invisible at any threshold you would dare set on a trim.

The reference comparison fails for a different reason, and a more instructive one: on a re-read
the honest answer to "which ayah does this end slice belong to?" is *this one*, because the
repeated span is this ayah's own words. It finds nothing on the population the corpus is made of.

## The detector: give the aligner the neighbours

Stop asking the aligner to explain bleed with reference text that cannot explain it. The
normalized decode is aligned **once** against `prev | this | next` concatenated, and bleed is
read off as the reference the alignment consumed outside this ayah's region.

Drag does not survive the change. Where bleed is real the neighbour's text pays a full
`MATCH_SCORE` per phoneme and the alignment takes it; where there is none the alignment can
still wander into a neighbour, but only over mismatches and skipped reference. So the two are
separated by **purity** — the share of consumed neighbour reference phonemes that matched
exactly — and not by span, and not by score:

| clip | leading matched/span | trailing matched/span | truth |
| --- | --- | --- | --- |
| 18:57 | 20 / 20 (1.00) | 26 / 26 (1.00) | bleed both ends |
| 18:18 | 13 / 13 (1.00) | 18 / 18 (1.00) | bleed both ends |
| 2:278 | — | 2 / 2 (1.00) | bleed, two phonemes |
| 12:48 / spk0002 | — | 2 / 2 (1.00) | bleed, two phonemes |
| 40:16 | 4 / 8 (0.50) | — | **drag** onto `يوم تتلاق` |
| 38:44 | — | 13 / 40 (0.33) | **drag**, 47 positions for +1.4 score |

Real bleed runs 0.80–1.00, drag 0.27–0.50. `MIN_BLEED_PURITY = 0.6` sits in the empty middle and
`MIN_BLEED_PHONEMES = 2` admits the shortest real bleeds. Widening the window to ±2 or ±3 ayat
changes no verdict on the pile; it only gives drag more reference to wander into (38:44's trailing
span grows 40 → 56 at the same purity), so ±1 stands, plus the **basmallah** as a second lead-in
candidate for a surah-opening clip.

The detector is torch-free and audio-free — it runs off `RejectRecord.predicted_phonemes`, the
decode the gate already stored, so the whole pile re-scores offline with no GPU. Reading the
reject sink no longer pulls in the audio stack either, which is what made that true.

### The labelled set, and one correction to it

`eval_fixtures/reject_bleed_labels.jsonl` restates the reviewer's free-text notes as per-edge
booleans, each with the `basis` it was read from, and carries every clip's decode so a re-score
needs only the fixture. It is the regression set: `test_bleed_detect.py` asserts P = R = 1.00
against it and that both baselines score strictly worse.

One label departs from the reviewer's note. **12:48/spk0009** is labelled *clean*, though its note
reads "Bleed into next ayah". Its decode ends on this ayah's own `تحصنۥن`, and its timed decode
runs to 15.17 s of a 15.33 s clip — 0.16 s of tail, nothing left untranscribed. The adjacent row
in the listening order, **12:48/spk0002** (ratio 0.681 against 0.688, near-identical decodes from a
different reciter), *does* end on `ثم`, the opening of 12:49, and carries no note. The note is on
the wrong row of the pair.

Two labels also go *beyond* their notes, in the same direction: 18:18 and 18:57 are labelled with
trailing bleed the detector finds (18 and 26 phonemes of the following ayah, at purity 1.00) and
the reviewer's note does not mention. Free-text notes are not exhaustive, so the decode is the
tiebreaker in both directions.

## Clipping

The phoneme-space boundary becomes a time through the CTC-timed decode
(`waqf_detect.collapse_with_times`), mapped from normalized phoneme indices back to raw decode
characters through `PhonemeNormalization.offset_map`. Every cut is clamped to the window between
the bleed and this ayah's own first/last decoded phoneme, so a mis-detection can only leave bleed
behind — it cannot eat recitation. Inside that window a **VAD silence** is preferred; the fallback
is a 50 ms pad short of the ayah (leading) or past the bleed's onset (trailing).

**A pause is there most of the time.** Of the cuts made, 18 of 21 leading and 15 of 20 trailing
landed inside a VAD silence. ADR-0016 decision 4 left pause coverage of a *re-read seam*
unmeasured and deliberately did not depend on it; at an **ayah boundary** — where a waqf is what
the reciter is doing — coverage is 33 of 41 cuts, 80%.

### Self-validation: every recut is re-decoded and re-gated

A recut is kept only if `match_ratio` **rises** and neither edge trim **grows**. The ratio must
rise because that is the whole claim: the bleed was inflating `query_phoneme_count`. The trims
must not grow because a cut landing inside the recitation shows up there first — the aligner
would start trimming real phonemes off the new edge. A cut that fails is discarded, so a bad
detection costs yield, never correctness.

**23 of 23 recuts passed**, on audio re-decoded through the model, not on the stored decode:

| band | clips | bleed | recut kept | audio clipped |
| --- | --- | --- | --- | --- |
| low (< 0.70, repeat) | 9 | 4 | 4 | 19.7 s |
| marginal (0.70–0.75) | 17 | 0 | — | — |
| clean (>= 0.75) | 19 | 0 | — | — |
| added shadda | 1 | 0 | — | — |
| no repeat (run < 5) | 25 | 19 | 19 | 175.3 s |
| **total** | **71** | **23** | **23** | **195.0 s of 1,603 s** |

The trims collapsing to `(0, 0)` on 21 of the 23 is the independent confirmation that the cuts
landed on the recitation's real edges — nothing told the aligner where they were.

## What this says about the corpus

**The low band is not mostly bleed.** This was the open question: if the 0.70-and-below clips were
low only because of bleed, the band would be a staging artifact rather than a store of long
re-reads. It is not. Four of the nine carry bleed at all, and after re-cutting:

| clip | ratio before | ratio after | Δ |
| --- | --- | --- | --- |
| 18:57 | 0.615 | 0.826 | +0.211 |
| 18:18 | 0.553 | 0.683 | +0.130 |
| 12:48 / spk0002 | 0.681 | 0.699 | +0.018 |
| 2:278 | 0.514 | 0.521 | +0.007 |

Two clips move materially and two barely move. Even after the cut, three of the four sit below
0.70 — their scores are dragged by the **re-read**, exactly as `docs/tadabur-ratio-floor.md`
argued (`corr(match_ratio, run/query_length) = −0.825`). Bleed and repeat length were confounded
in the same denominator; separating them leaves the repeat as the dominant term. The decision to
drop the ratio floor stands, and now stands on a measurement rather than on an inference.

**The rejects that carry no repeat are a different population entirely.** 19 of the 25 `run < 5`
rejects are bleed, and they are *mostly* bleed: 20:1 goes 0.081 → 0.875, 74:5 0.155 → 0.944,
96:3 0.245 → 0.929. These are correct recitations of short ayat that the gate rejected because the
staged clip contained two or three ayat's worth of audio. They are not ADR-0016 corpus material
(no re-read), but the same recut would recover them as ordinary training data — a yield question
for the ADR-0001 filter, noted here and not pursued.

## Truncation, the mirror case

The same alignment answers it: `uncovered_tail` is the count of this ayah's reference phonemes
falling after the decode's last matched position. Scored against the labelled nine at a 5-phoneme
bar it is exact (P = R = 1.00, one positive — 40:16, the clip the reviewer heard stop early).

Four of the 71 rejects are truncated, and every one stops at a plausible waqf:

| clip | stops before | uncovered |
| --- | --- | --- |
| 40:16 | `لواحد لقههار` | 13 |
| 3:14 / spk0180 | `وللاه عندهۥ حسن لمءاب` | 22 |
| 3:14 / spk0024 | `وللاه عندهۥ حسن لمءاب` | 22 |
| 35:32 | `ذالك هو لفضل لكبۦر` | 19 |

Two different reciters stop at the identical point in 3:14, which is the tell: this is a reciter
pausing at a waqf, not a clip cut short at random.

**Keep them, with a shortened covered range — do not drop them.** There is nothing to cut: the
audio is all recitation, and it is correct recitation. ADR-0016 decision 1 already scopes every
assertion to the recited word range obtained from the alignment, so a truncated clip needs no new
machinery — only for `uncovered_tail` to be carried so the range is shortened rather than the
clip's final words being read as skipped. Dropping them would cost 5.6% of the pile to avoid a
problem the oracle already handles. `RecutRecord.uncovered_tail` carries it.

## Interfaces

`RecutRecord` (`bleed_run/recuts.jsonl`, one row per reject) carries `recitation_start_s` /
`recitation_end_s` shaped exactly like `ClipStatus`'s fields — a rejected clip has no `ClipStatus`,
so the span travels here instead. The span is always usable: the recut when it was accepted, the
whole clip otherwise, so a reader never branches on `accepted` to get a valid span. #64's stager
and Muraja's transcriber read those two fields.

The phoneme-space-boundary-to-time conversion is shared with #64's excision work as the issue
asked: both go through `collapse_with_times`, and `normalized_onsets` is the piece that maps the
normalized phoneme indices an alignment speaks in back onto it.

## Limits

- **One shard.** 71 rejects, 9 adjudicated clips. Prevalence by band is a shard-20 measurement,
  not a corpus one; the detector's thresholds sit in an empty middle wide enough that this is not
  the fragile part, but the *rates* will move.
- **Bleed the model did not transcribe is invisible here, and harmless here.** A detector reading
  the decode cannot see audio the decode missed. That bleed also does not inflate
  `query_phoneme_count`, so it does not drag `match_ratio` — but it *is* still in the audio the
  replay hears. No such case was found in the nine (every clip's decode runs to within 0.64 s of
  its end, and none of the 71 leaves more than 0.88 s), so this is a stated limit, not an
  observed one.
- **Ambiguous repeated ayat.** Where consecutive ayat share text — al-Kafirun is the extreme, and
  109:5 is in this pile — "bleed from the neighbour" and "re-read of this ayah" are not
  distinguishable from text alone. Its recut was accepted (0.315 → 0.436) but its trailing trim
  stayed at 20; the re-gate contained the ambiguity rather than resolving it.
- **Decode nondeterminism.** Re-decoding the 71 clips reproduced 68 stored decodes exactly; the
  three differences are one madd length and one diacritic from bf16 batching, and none moves a
  detector verdict. Worth knowing before treating a stored decode as a fingerprint.

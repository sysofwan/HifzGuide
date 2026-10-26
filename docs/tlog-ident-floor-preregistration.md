# Re-pre-registering the "is this recitation?" floor (S5b, #204)

**Dated 2026-10-26. Post-look re-pre-registration.** This document replaces, for verdicts only,
the per-length-bin floor of S5 (`docs/tlog-ident-preregistration.md`). It is triggered by S5's
own frozen stop clause: the pool run let 181/445 Freesound recordings (40.7%) pass as recitation
against a 1% limit, so S5 recorded `floor_status: "stopped: noise 40.7% > 1%"` and the floor was
not carried downstream. Nothing in S5's report, its decodes or its pools is edited or re-scored
by this document; the binned floors stay in `ident_calibration.json` as history.

This is a **post-look** document, in the sense of the frozen hold-out text: L_min and DELTA were
chosen after looking at the look decodes and the 445-recording pool. The fresh confirmation sets
drawn under §3 are the only thing that makes the rule trustworthy, and the order of work below
is binding. The values are fixed here and are never selected on any confirmation data.

## 1. Parents (never edited) and their SHA-256 at this document's commit

- `docs/tlog-holdout-preregistration.md` —
  `204fe7f980488aff73402c56f1729f7f17f7c9471ea0664f6365bb8870b62086`
- `docs/tlog-ident-preregistration.md` (S5, 2026-10-25, pre-outcome) —
  `a383fcfdbd375c9e6c052d8032788a8eda69bd2d9fbd12fcd3b921b1fbb6a3ab`
- `docs/tlog-ident-bu-amendment.md` (owner-approved post-look b_u erratum) —
  `579f9ba1bb751e65663e3a6f477b6d39c8503c94811050951e1338cc71aa76c8`

These three documents are frozen. Any amendment is a new dated document citing its parent by
SHA-256.

## 2. The rule (fixed; never selected on any data in this issue)

A decode with speech present is **`non_recitation`** when **either**:

- its normalised decode (`normalize_phonemes`, as S5 pins) is shorter than **L_min = 8**
  characters — an empty decode counts; **or**
- `real_best − decoy_best < δ = 0.30`, where both scores come from **the same shortlisted
  search** of that same decode, as pinned in `docs/tlog-ident-preregistration.md` (shortlist
  k = 20, edlib 1.3.9.post1 `HW` infix mode, decoys by grapheme-cluster reversal, the pinned
  tie-break).

There are no length bins. The binned floors and the `161-` question are retired for verdicts;
they remain in the report and the page as history. Every identification record carries
`floor_rule {path, sha256 of this document}`, so no row can silently mix the two regimes.
`floor_ok` / `id_floor_ok` keep their names and now mean this rule's verdict.

## 3. Already-seen values (disclosed, not fresh evidence)

These are post-look values on data the rule was chosen against. They are disclosed here and
labelled as such; they are **not** bars.

| set | value |
|---|---|
| Tarteel look kept | 99.78% |
| look tail (`cer_wav ≥ 0.2`) kept | 99.72% |
| 445-recording Freesound pool passing | 0/445 |
| real-mistake clips passing | 52/52 |

Neighbourhood already inspected: every L 6–10 × δ 0.20–0.40 gives 0–3 noise recordings passing
and ≥ 99.4% look kept. This document does not add a second neighbourhood to select from.

## 4. The fresh sets, drawn and committed before any of them is decoded

All draws use salted SHA-256 ranks with the salt **`tlog-s5b-floor-confirm-v1`**. The sizes below
are fixed before drawing. A draw is a pure function of the source rows, the guard, the salt and
the speaker cap: the fetch route only supplies bytes.

### 4a. Freesound noise: exactly 1,000 recordings

- **Source:** `obadx/freesound-commercial-50k-noise-only` at
  `a5f48ab83b64ba1d35ee4ec363dae75d4e2c69d3`, the source and revision of the S5 pool
  (`tools/training/perturbation.py`).
- **Shards:** exclude shards **0 and 52** (the `nonpro_gap` pool) and **26 and 78** (the
  augmentation pool). Rank the remaining 100 of the 104 shards by salted hash
  (`sha256("tlog-s5b-floor-confirm-v1:shard:<name>")`) and take them in that order until at least
  1,000 recordings are eligible; **stop and report** the number if 10 shards are exhausted first.
  Do not lower the bar. Each shard is streamed and deleted after use.
- **Eligibility:** `perturbation.pool_eligible` (decodes, ≥ 1.0 s, not below −80 dBFS). In
  addition, drop any recording whose `freesound_id` or `samples_sha256` appears in either
  committed pool (`nonpro_gap/noise_pool.json`, `augmentation/noise_pool.json`).
- **Selection:** the first 1,000 eligible recordings by salted rank of `freesound_id`
  (`sha256("tlog-s5b-floor-confirm-v1:fs:<id>")`).
- **Commit:** `tools/tadabur/tlog/ident/confirm/freesound.json` with, per recording,
  `freesound_id, licence, username, samples_sha256, seconds, shard`; plus the full 100-shard
  ranking, the drawn shard prefix, the counts left out per shard by reason
  (`undecodable`, `ineligible`, `in_nonpro_gap_pool`, `in_augmentation_pool`) and the licence
  mix. No audio is committed.

### 4b. Common Voice non-Quran Arabic speech: exactly 500 clips

- **Release:** `cv-corpus-27.0-2026-09-11`, from Mozilla Data Collective (CC0-1.0), **validated
  clips only**, no duration filter. Record the release name and each archive's SHA-256.
- **Quran-text guard, text only and before the draw:**
  - Reference text: `data/imlaei.json` (modern Imlaei spelling per word), **not** the Uthmani
    text. Normalise each word with `normalize_imlai` (`tools/generate_quran_db.py`), dropping the
    dagger alef (U+0670), tatweel and every non-Arabic-letter character, and drop verse-number
    tokens.
  - Build 4-grams from **consecutive words within a surah**, across ayah boundaries, so a quote
    spanning an ayah is caught.
  - Exclude a CV sentence if any 4 consecutive normalised words appear contiguously in those
    n-grams. This catches the basmala, quoted ayat and similar. Record the number excluded per
    locale.
  - Applies to every Arabic-script locale drawn: `ar`, `ur`, `fa`, and `ckb` if substituted.
- **Speakers:** rank clips by `sha256("tlog-s5b-floor-confirm-v1:cv:<locale>:<path>")` and take
  them greedily in rank order: pass 1 at most **1 clip per `client_id`**; if pass 1 ends short of
  500, pass 2 at at most **2 per `client_id`** over the clips not yet taken. Maximum cap 2. If
  still short, stop and report the number; do not substitute for `ar`. Record the cap reached and
  the per-cap counts.

### 4c. Common Voice non-Arabic speech: 100 clips each from five locales

- Locales: **`en`, `ur`, `fa`, `tr`, `id`** — the app's likely users, with Urdu and Persian the
  hard cases (Arabic-script and heavy Arabic borrowing). Same source, release, guard (for `ur`,
  `fa` and any `ckb` substitute), speaker rule and salt as 4b.
- **Substitutes**, recorded here before the draw as a contingency with a mechanical trigger:

  | primary | substitute |
  |---|---|
  | `en` | `es` |
  | `ur` | `hi` |
  | `fa` | `ckb` |
  | `tr` | `sw` |
  | `id` | `bn` |

  A substitute fires only if its primary has fewer than 100 eligible clips at cap 2, computed
  from `validated.tsv` after the guard and before any ranking is looked at. The substitute must
  itself reach 100 at cap 2, otherwise stop and report. At most one substitution per locale; no
  other discretion. Eligible counts are recorded for every primary either way.
- **Commit:** `tools/tadabur/tlog/ident/confirm/common_voice.jsonl` (one line per clip:
  `arm, locale, path, client_id, sentence, audio_sha256, duration_ms, release`) and the counts
  sidecar `common_voice.json` (salt, release, the archive pins, per-locale eligible/excluded/
  examined counts and the cap reached). No audio is committed.

### 4d. Archive pins (Mozilla Data Collective, release 27.0)

Every archive below is `.tar.gz`, CC0-1.0, pinned by dataset ID, filename, SHA-256 and size from
the MDC API before any download. The six primaries are downloaded; the five substitutes are
fetched only if their substitution fires. Sizes are exact bytes.

| locale | role | dataset id | filename | sha256 | bytes |
|---|---|---|---|---|---|
| `ar` | primary | `cmu5wom3z00ceo107guxtg4xa` | `common-voice-scripted-speech-27-0-arabic-b05a34b5.tar.gz` | `dbde785fa6e721202bf70d8aecd940e763703b84c573e043fa71ac1e1665913a` | 3,542,436,951 |
| `en` | primary | `cmu5jplf300nwmh07iqvk9leo` | `common-voice-scripted-speech-27-0-englis-e4a13e08.tar.gz` | `2fbc16e9811e0a727c7fde5c71158a9295a2aa5e26f9ec21b305f0a8368a10c7` | 94,956,262,763 |
| `ur` | primary | `cmu5x1w3400cyo107st8cgsoq` | `common-voice-scripted-speech-27-0-urdu-19469962.tar.gz` | `12ad5f359fd94805db1bdaeb1441eb35720221f3c99229034069375fa246d990` | 6,218,878,938 |
| `fa` | primary | `cmu5xazgy00finq074vaqo1b3` | `common-voice-scripted-speech-27-0-persia-9d86e98b.tar.gz` | `f71fd362846433376b529d3f2ce4258a4075d9da8fe2b0f4b12465c12672a65d` | 11,236,660,956 |
| `tr` | primary | `cmu5wkah500c2o10719lem8m8` | `common-voice-scripted-speech-27-0-turkis-b14760fe.tar.gz` | `10becdc2fb112f8cf7ffd983e269dd2013aef5cb3f747b7e5f455146a21d6c8e` | 2,992,390,702 |
| `id` | primary | `cmu5wcato00cknq07yjnu14e0` | `common-voice-scripted-speech-27-0-indone-27d20112.tar.gz` | `d25a6b734ed0e317515eb1264fa03a925a4638aa12f275abb31637e8504d28fa` | 1,531,290,920 |
| `es` | substitute | `cmu5kitx700pemi073feoiqqu` | `common-voice-scripted-speech-27-0-spanis-8d20e589.tar.gz` | `b7c1f303ebe3c4a637afdd3d31c5aaa3da131a893ce77169db182068f342e4c1` | 51,943,162,413 |
| `hi` | substitute | `cmu5vlext009to107ev1s9lk8` | `common-voice-scripted-speech-27-0-hindi-25c8439c.tar.gz` | `8d08563ef25a9796a0f12f8c8c6d4190b976ed2bf95d9c8082c1ce8a12466513` | 573,120,742 |
| `ckb` | substitute | `cmu5wr35b00donq07jqzj68wa` | `common-voice-scripted-speech-27-0-centra-cd0ad022.tar.gz` | `88c7f0322898b5402146ba296cf19a9b4e07297ae33667227a5b9afb5259f3b4` | 3,916,538,045 |
| `sw` | substitute | `cmu5sgles002zo1071nphgtby` | `common-voice-scripted-speech-27-0-swahil-a1aabcc5.tar.gz` | `45811dfd9c6146513e6ef350c5d60b7bc97ce55411e553e204ac082b31bdc2ba` | 22,427,928,356 |
| `bn` | substitute | `cmu5kut0d00pomh07yay2f34r` | `common-voice-scripted-speech-27-0-bengal-dca36a3e.tar.gz` | `3cdebe9005bef6b53d829e0f544297c04df0b1b801c170b1be23adfc917783a6` | 26,681,655,257 |

### 4e. Fetch routes

- **Primary:** the full archive, downloaded on the Mac with `curl -C -` into NAS scratch
  (`/Volumes/NetBackup/HifzGuide/scratch/common-voice/`), SHA-256 checked against the pin above.
  Draw there, commit the manifests, then copy only the drawn clips to the GPU box.
- **Fallback**, only if the transfer fails: two streaming passes with nothing stored — pass 1
  extracts only the `.tsv` files, draws and commits the manifests (without audio hashes); pass 2
  extracts the drawn clips, hashes and stages them. Both routes read the same rows and apply the
  same pure draw rule, so they produce the identical selection. A one-pass reservoir is rejected:
  it would be a second draw rule.

## 5. Decode modes (base teacher only)

- **Model:** `obadx/muaalem-model-v3_2` @ `01a1ef9fbe40d144ef845101e89ff924aed3fef5`, under the
  GPU lock on the GPU box.
- **Freesound:** `stream_b0`, the mode the 445-recording pool was decoded in
  (`NOISE_DECODE_MODE` in `tadabur.tlog_ident`), so the fresh set confirms the same condition
  those values were read from.
- **Common Voice:** whole-clip, through `tadabur.audio:decode_to_mono_16k_with_rate_and_decoder`
  (16 kHz mono; the same loader and resampler S6/S7 use for tlog rows).
- **No C95, student or candidate decode anywhere.**
- The decode script refuses to run unless this document and both manifests are committed and
  clean in git, and every decode store records their SHA-256.

## 6. Bars, reports and the stop

| bar | limit |
|---|---|
| fresh Freesound pass rate | ≤ 1% (≤ 10/1,000) |
| CV non-Quran Arabic pass rate | ≤ 1% (≤ 5/500) |
| CV non-Arabic pass rate | ≤ 1% (≤ 5/500) |
| real-mistake clips passing | 52/52 |
| Tarteel look retention (frozen bar) | ≥ 90% |

Each pass rate is reported with a Wilson 95% interval. **Reported, not bars:** the 445-recording
pool's rate; look tail retention; the non-Arabic rate per language; for each set, the share
caught by L_min against the share caught by δ; and the passing items' decodes and best spans.

The report records `floor_status: "confirmed"` or `"stopped: <which bar, value>"` beside this
document's SHA-256. **Stop:** if any bar fails, the numbers are posted on #173 and #190 for the
owner. L_min and δ are not touched and nothing is redrawn; the status stays `stopped`.

## 7. Downstream, only if confirmed: one re-derivation

Under the confirmed floor, once: `b` by its frozen rule, and `b_u` by the owner-approved
precision rule (`docs/tlog-ident-bu-amendment.md`). Both drop their `provisional` mark and are
posted on #173. `b_u` is not final until the owner approves the margin definition at the S5b
check-in.

## 8. Character-level margin check (recorded, never a selection step)

The character-level margin competitor runs on the **look set only**, after this document and
before the final `b_u`. It reuses the pinned word-span competitor: for each decode, prune a start
only when its character-level score is strictly greater than `winner.score − pinned margin`, then
attribute the best path. The result is written to
`tools/tadabur/tlog/ident/oracle_margin.json` and recorded as
`{b_u_oracle, clips_m_changed, assignments_changed_at_b_u, artifact}`. It never feeds `b_u`, and
the bars are always derived from the pinned margins. Its consequence is mechanical: `final` only
when it changed neither `b_u` nor a look assignment at `b_u`, otherwise `owner`.

## 9. What this does not test

Recitation-like non-Quran material — nasheed, adhan, or du'a in a Quranic style — is **not** in
any set here. The S6 audit is the field check for that. If the owner hears at least 3 of the 25
audited `non_recitation` items as real recitation, it is reported for a later dated revision; the
same if non-recitation turns up in `both`. Nothing is retuned in this issue.

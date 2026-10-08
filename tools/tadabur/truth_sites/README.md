# Truth sites (#79)

A **truth site** is one human-labelled position in one piece of Tadabur audio: what the
mushaf prescribes there and what the reciter actually said. It holds no model output.
[ADR-0011](../../../docs/adr/0011-transcription-fidelity-and-tashkeel-abstention.md) judges
every model against these sites, never against the teacher or the mushaf.

The files in this directory are committed source-of-truth (AGENTS.md, *Data & assets*). Each
`*.jsonl` needs its own `!` entry in the root `.gitignore`. The schema and the validating
loader are in [`tools/tadabur/truth_sites.py`](../truth_sites.py).

| file | what it is |
|---|---|
| `waqf_boundaries.jsonl` | the adjudicated waqf boundaries, converted to tashkeel sites |
| `waqf_boundaries.summary.json` | where each of the 2,050 boundary rows went (a site or an exclusion) |
| `p35_fixtures.jsonl` | the P3.5 should-accept / should-reject fixtures, re-located on today's segments (#83) |
| `p35_fixtures.relocation.jsonl` | every fixture's re-location: outcome, segment span, realized reference, base decode, site ids |
| `p35_fixtures.summary.json` | kept / dropped per bucket and verdict, sites per stratum, the decode fingerprint |

The staging fields of every file come from the staged-clip registry,
[`../staged_audio/clips.jsonl`](../staged_audio/README.md), which also carries each clip's
canonical reciter id.

## Schema

One JSON object per line, keys sorted, UTF-8. Every field is always present. A field that
is not known yet is `null`. It is never guessed.

| field | type | when | meaning |
|---|---|---|---|
| `site_id` | str | now | stable, unique id. Waqf sites use `waqf_boundary:<clip>#<boundary_index>` |
| `source` | str | now | `p35_fixture`, `waqf_boundary`, `new_audit` or `synthetic_edit` |
| `assumes_competent_reciter` | bool | now | `true` when the human judged something else and the mark holds only if the recitation was correct (the weaker label) |
| `audio_filename` | str | now | Tadabur `audio_filename` of the whole clip |
| `shard` | int \| null | **#83** | full-config parquet shard (0–384) the clip is re-downloaded from |
| `start_sample` | int | now | first sample of the item in the clip (16 kHz). 0 for a whole clip |
| `end_sample` | int \| null | **#83** | end sample (exclusive) of the item. For a whole clip, its length once staged |
| `audio_sha256` | str \| null | **#83** | SHA-256 of the staged 16 kHz mono PCM_16 WAV's bytes (`truth_sites.audio_sha256`) |
| `surah_ayah` | str | now | `"surah:ayah"` |
| `reference` | str | now | the item's **realized reference**: the phonetizer's output for what was recited, with waqf/wasl as recited |
| `reference_index` | int | now | index in `reference` of the **carrier letter**, the consonant the mark under test sits on (for shaddah, the first of the doubled pair) |
| `mark` | str | now | what is under test: `fatha`, `damma`, `kasra`, `sukun`, `shaddah`, or a soft pair labelled as `tadabur.phoneme_sifat.soft_pair_contrasts()` spells it, in codepoint order (`ت↔ط`, `ح↔ه`, `ذ↔ز`, `س↔ص`, `ض↔ظ`, `ق↔ك`) |
| `prescribed` | str | now | what the mushaf prescribes there (table below) |
| `heard` | str | now | what the human heard (table below). `unclear` and `pending` leave the denominator |
| `stratum` | str | now | the sampling stratum the site was drawn from |
| `stratum_population` | int | now | how many positions that stratum holds, so sampled estimates can be weighted (weight = population / sites in the stratum) |

`shard`, `end_sample` and `audio_sha256` are the **staging fields**. #83 fills them when it
re-stages the audio. All three are `null` (not staged yet) or all three are set (staged).
The loader rejects a mix.

Every mark may also be heard as `unclear` (the listener could not tell) or `pending`: the
site is defined and staged, but no **site-level** verdict exists yet. #83 added `pending` for
the P3.5 clip-level rejects (below), which do not say what was said at a given site; the
listening session (#87) replaces it with a verdict. A scorer treats `pending` like
`unclear`: out of every denominator.

| `mark` | `prescribed` | `heard` |
|---|---|---|
| `fatha` / `damma` / `kasra` / `sukun` | the mark itself | `fatha`, `damma`, `kasra`, `sukun`, `unclear` or `pending` |
| `shaddah` | `held` (a geminate in the mushaf) or `not_held` (a single consonant: the site of an *added* shaddah) | `held`, `not_held`, `unclear` or `pending` |
| a soft pair `a↔b` | `a` or `b` (the mushaf's letter) | `a`, `b`, `unclear` or `pending` |

`load_truth_sites(path, audio_dir=None)` rejects:

- **malformed rows**: invalid JSON, a missing or unknown field, or a wrong JSON type (`true` is not an int, `1.0` is not an int);
- **unknown sources or marks**, and a `prescribed` / `heard` the mark cannot take;
- **missing provenance**: an empty `audio_filename`, a `null` required field, partial staging fields, a bad shard, span or checksum format;
- **a reference index that does not carry the mark**: the carrier is not a consonant, a haraka site is not followed by that haraka, a sukun site is followed by a haraka, a `held` shaddah site is not the first of a doubled consonant, a `not_held` one is not a single consonant, or a pair site's carrier is not the prescribed letter;
- **inconsistent files**: duplicate `site_id`, sites on one clip that disagree on its `shard` or `audio_sha256`, sites on one item (`audio_filename`, `start_sample`) that disagree on its `end_sample`, `surah_ayah` or `reference`, or a stratum whose population is not one value at least as large as its site count;
- **a checksum mismatch** when `audio_dir` is given. Every item's `<audio_dir>/<audio_filename>` must hash to its recorded `audio_sha256`, and an item with no checksum fails, because it cannot be verified.

`write_truth_sites` runs the same checks before it writes anything, then writes atomically.

## `waqf_boundaries.jsonl`: the adjudicated waqf boundaries

Source: the 2,050 frozen rows in `../waqf_event_fixtures/waqf_events.{calibration,test}.jsonl`
(126 clips, human verdicts `waqf` 179 / `wasl` 1,857 / `mid_word_closure` 14). Regenerate it
deterministically, from `tools/` with `quran-transcript` installed:

```bash
python -m tadabur.waqf_truth_sites
```

**Item and reference.** Each clip is one item: `start_sample = 0`, `end_sample` the staged
clip's length, and `shard` / `audio_sha256` from the staged-clip registry (#83). A clip the
registry lacks would keep all three `null` and be listed under `clips_not_restaged` in the
summary; none does. The boundary times in the fixture are whole-clip times, so no
re-segmentation is needed. The realized reference splits the clip's words at every human
`waqf` and phonetizes each recited run on its own (Hafs, `generate_phonemes.HAFS_MOSHAF`):
the terminal word in waqf form, the rest in wasl. The runs are joined by a space. A re-read
shows up as a run that restarts at an earlier word.

quran-transcript 0.5.2 gets one pausal form wrong: its `MaddAlewad` step turns *every* final
tanween fatha into fatha + alif, so `رَحْمَةًۭ` at waqf comes out `رَحمَتَاا` instead of
`رَحمَه`. The converter rewrites a run-final `ةً` as `ةَ` before phonetizing
(`pausal_taa_marbuta`), which yields ه with sukun. Every other pausal ending in the data was
checked against the rule it must follow (taa marbuta with any tanween → ه; tanween kasra /
damma → sukun; tanween fatha → madd al-iwad `َاا`, after a hamza too; madd endings → long
vowel; any other consonant → sukun) and is correct. The same phonetizer bug affects every
other realized reference in the repo that ends a waqf segment on `ةً`.

**Site.** The site is the boundary word's final mark: the haraka after its last realized
consonant, or sukun when there is none. If the word ends in a madd letter, the mark sits on
the letter before it (`عَلَى` → `عَلَ` / `عَلَاا`).

- `waqf` and the word ends in a consonant → **sukun at a pause**,
  `assumes_competent_reciter: false`. The pause the human heard is the evidence.
- `waqf` and the word ends in a long vowel (a madd letter, or tanween fatha, which becomes
  `اا` at waqf) → the haraka before the madd, `assumes_competent_reciter: true`. A pause on
  a long vowel produces no sukun, so the verdict says nothing about that mark.
- `wasl` → the word's wasl ending: a haraka, or the sukun the mushaf writes there,
  `assumes_competent_reciter: true`. The human heard continuation, not the mark. This is
  the weaker label: it holds only if the reciter recited correctly.

`prescribed == heard == mark` on every site. These labels are all correct recitation; none
is a real mistake. Strata are `waqf_boundary:waqf` and `waqf_boundary:wasl`. The 126
reviewed clips were labelled in full, so the strata are a **census** of them: the
population equals the site count, every weight is 1, and the clips themselves were not
sampled in a way that can be weighted back to the corpus.

### Counts

Sites by stratum and mark (1,792 sites on 123 clips; 1,657 assume a competent reciter):

| stratum \ mark | `damma` | `fatha` | `kasra` | `sukun` | total |
|---|---|---|---|---|---|
| `waqf_boundary:waqf` | 11 | 24 | 0 | 135 | 170 |
| `waqf_boundary:wasl` | 364 | 777 | 299 | 182 | 1622 |

All 1,792 sites have `source: waqf_boundary`. Where each fixture row went:

| fixture verdict \ outcome | `closure_merged` | `final_letter_assimilated` | `inconsistent_clip` | `mid_word_closure` | `phonetizer_unsupported` | `waqf_boundary:waqf` | `waqf_boundary:wasl` | total |
|---|---|---|---|---|---|---|---|---|
| `mid_word_closure` | 0 | 0 | 0 | 14 | 0 | 0 | 0 | 14 |
| `waqf` | 15 | 0 | 4 | 0 | 5 | 155 | 0 | 179 |
| `wasl` | 8 | 143 | 41 | 0 | 28 | 15 | 1622 | 1857 |

### Why this is not 179 sukun-at-pause and 1,857 wasl sites

ADR-0011 counted the fixture's **verdicts**. A verdict is a site only when it fixes one
tashkeel mark on one letter of a reference that can be built:

- **Closure candidates are not word edges.** The detector also proposed VAD silences inside
  words (`predicted: mid_word_closure`), placed on a word by the phoneme alignment. The
  regular edges' times are interpolated, so where a closure sits in time says nothing about
  which edge it is. A closure the human called `waqf` or `wasl` is therefore merged into the
  regular edge after **the same word** (`closure_merged`: 15 waqf, 8 wasl), and a `waqf`
  there makes that edge a pause. That is why 15 `wasl` rows land in the `waqf` stratum. A
  closure whose word has no regular edge (a stop after the ayah's last word) is an edge of
  its own. A `waqf` closure on a word a re-read recites twice cannot be placed on either
  pass. The clip would then be excluded whole (`closure_ambiguous`), rather than continue
  the reference through a known pause, but no clip in this data has one.
- **35 pauses fall on a long vowel** and are haraka sites, not sukun (see above). So the
  `waqf` stratum is 135 sukun + 35 haraka.
- **143 wasl boundaries have no word-final mark of their own** (`final_letter_assimilated`).
  The last letter merges into the next word through idgham (`أَن نَّمُنَّ` → `ءَننننَمُ…`,
  `أَن يَ` → `ءَييَ`) or is nasalized into it through ikhfa / iqlab (`مِّن قَوْمِهِۦ` → `مِںںںقَومِه`).
  The haraka or sukun under test does not exist there.
- **3 clips yield no reference.** In `spk0192_S1_A178` the edges run 0–7 within 0.6 s and
  then restart at word 0 with no pause before it, so they are not one recitation
  (`inconsistent_clip`, 45 rows). `spk0177_S26_A88` and `spk0345_S10_A52` each have a recited
  run that ends in waqf on a leen letter followed by the final consonant (`شَىْءٍ`,
  `إِلَيْهِ`). quran-transcript cannot phonetize that ending, so these clips are
  `phonetizer_unsupported` (33 rows).
- **14 `mid_word_closure` verdicts** are not boundaries and are excluded, as the issue specifies.

## `p35_fixtures.jsonl`: the P3.5 fixtures, re-located (#83)

Source: the 209 P3.5 audit labels in `../eval_fixtures/should_{accept,reject}.jsonl` (174
accept, 35 reject). Each id `<clip>__seg<n>.wav` names segment `n` of a whole clip, and each
row judged one contrast bucket on it. The segment boundaries were lost with `audit_run/`, so
the labels are re-located on **today's** segments. Regenerate, from `tools/`:

```bash
# On the GPU box: segment the staged fixture clips and decode them with the base teacher.
python -m tadabur.resegment --registry tadabur/staged_audio/clips.jsonl --use p35_fixture \
    --audio-dir stage/clips --out-dir stage/seg_p35
# Anywhere (torch-free): apply the re-location rule and write the three files.
python -m tadabur.p35_truth_sites --seg-dir stage/seg_p35
```

`tadabur.resegment` runs today's segmentation (`tadabur.segment_score`: the recitation VAD,
pause-to-word placement, the drop rules), with every decode made through
`training.decoding.Decoder` by the base teacher (`obadx/muaalem-model-v3_2`, bf16 weights,
whole spans, batch size 1; the fingerprint is in the summary). Its references take the
run-final word through `pausal_taa_marbuta` (#79, #100), so no segment ends in `تَاا`.

**The re-location rule** (`tadabur.p35_truth_sites.relocate`). A fixture keeps its label
only if its bucket names a contrast, segment `n` of its re-staged clip exists and survives
the drop rules, and the base decode of that segment still shows **the labelled contrast**
against the segment's realized reference (`contrast_attribution.contrast_sites`): the
soft-pair substitution for a pair bucket, a gemination mismatch for `shadda`. Each
occurrence is one site, on its carrier: the substituted letter (a geminate substituted
whole is one site, on its first half), the first of a doubled pair the decode left single
(`prescribed: held`), or the single consonant the decode doubled (`prescribed: not_held`).
An occurrence whose carrier the schema cannot hold (a folded ghunna noon) is skipped.

**What the verdict says at the site.**

- `accept`: the labeller judged the clip acceptable for that contrast, so the mushaf's
  letter (or gemination state) was said at each occurrence: `heard = prescribed`.
- `reject`: a **clip-level** verdict. It does not establish that the other letter was said at
  a given site, and several carry notes such as "Not hafs", "Segmentation issue" or "Unclear
  reading". No `heard` value is invented: re-located reject sites are `heard: pending`,
  waiting for the site-level re-adjudication the listening session (#87) gives all nominal
  rejects (acceptance rules §8).

Every site is `assumes_competent_reciter: false`. Strata are `p35_fixture:<bucket>`, each a
**census** of the re-located fixtures (population = site count, weight 1). These are the
targeted safeguards of the acceptance rules (§1), never pooled into headline rates.

### Kept and dropped, per bucket

| bucket | verdict | fixtures | kept | `contrast_absent` | `segment_missing` | `segment_dropped` | `contrast_not_expressible` | `marginal_no_site` | sites |
|---|---|---|---|---|---|---|---|---|---|
| `ذ↔ز` | accept | 33 | 26 | 5 | 2 | 0 | 0 | 0 | 28 |
| `ذ↔ز` | reject | 3 | 2 | 1 | 0 | 0 | 0 | 0 | 2 (pending) |
| `ت↔ط` | accept | 13 | 9 | 2 | 1 | 1 | 0 | 0 | 9 |
| `ت↔ط` | reject | 1 | 1 | 0 | 0 | 0 | 0 | 0 | 1 (pending) |
| `ض↔ظ` | accept | 28 | 25 | 1 | 1 | 1 | 0 | 0 | 25 |
| `ض↔ظ` | reject | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 2 (pending) |
| `ق↔ك` | accept | 27 | 27 | 0 | 0 | 0 | 0 | 0 | 27 |
| `ق↔ك` | reject | 3 | 3 | 0 | 0 | 0 | 0 | 0 | 3 (pending) |
| `س↔ص` | accept | 28 | 28 | 0 | 0 | 0 | 0 | 0 | 28 |
| `س↔ص` | reject | 2 | 2 | 0 | 0 | 0 | 0 | 0 | 2 (pending) |
| `ح↔ه` | accept | 4 | 3 | 0 | 0 | 1 | 0 | 0 | 3 |
| `shadda` | accept | 18 | 17 | 0 | 0 | 0 | 1 | 0 | 17 |
| `shadda` | reject | 12 | 12 | 0 | 0 | 0 | 0 | 0 | 13 (pending) |
| `marginal` | accept | 23 | 0 | 0 | 0 | 0 | 0 | 23 | 0 |
| `marginal` | reject | 12 | 0 | 0 | 0 | 0 | 0 | 12 | 0 |
| **total** | | **209** | **157** | **9** | **4** | **4** | **1** | **35** | **160** |

157 of the 174 contrast fixtures are kept: 135 of 151 accepts and 22 of the 23 nominal
rejects. They give 160 sites: 137 heard as the mushaf (120 soft-pair; 17 shaddah, 16 `held`
and 1 `not_held`), and 23 `pending` (10 soft-pair; 13 shaddah, 7 `held` and 6 `not_held`). The one reject lost
(`ذ↔ز`, "Unclear reading") no longer shows the substitution. Every staged fixture clip
was re-staged; none was lost to re-staging. `ح↔ه` has accepts only and `ت↔ط` one reject, so
neither pair can support a per-pair claim (PRD #77).

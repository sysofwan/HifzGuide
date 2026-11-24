# Follow-along audit: the App arm on placeable phone audio (#309)

Generated from `app_arm/summary.json` by `python -m tadabur.followalong_audit report`; never edited by hand.

## Population

5,742 claim-gated confirm.1/confirm.2 gating clips — the ones #248's exclusion leaves (its ayat the phonetizer cannot read have no reference to grade).
5,742 of 5,742 clips produced a word table (100.0%).

## Run identity

- Muraja harness commit `028bc513eeb233caa7240a5c37c423da59f3d9d7`; assets release `v1.0.7`.
- App-arm latency 40 ms per inference; P2 seed 319; P3 seed 309.
- Driver sha256 `d58dbd066ddaea4640f79b561b307c35e1f1e8ab9e6f1447b23f085833192108`; harness binary sha256 `6913c94384950653c0fa7889eb320b63d16c22424244d38d12d57b6af0aa7bef`.
- `audio_manifest.jsonl.gz` sha256 `1162aa8b4adba6c0a8b7a0d053bcc9e5dcd3c71ccd820e7f88af819676d57982`; `word_grades.jsonl.gz` sha256 `6064be7082341690db9534f987b636c0e6bd482abf66aa1bd708dd9473dbb3b2`.
- Determinism: the 25 seeded clips ran twice and all 25 driver payloads were byte-identical in this sample; a sample bounds, it does not prove, bit-identity (`app_arm/determinism.json`).
- Known counterexample (`app_arm/payload_probe.json`): `41_29_undefined_ItjrgCt.flac#0` ran 20 times at 40 ms: 20 succeeded, 0 failed; the successful runs' decode took 2 distinct payloads (4× `7e4e27d387ee`, 16× `914b1c85ff94`), and their word table was identical — the decoder is not bit-stable, and diagnosing that is S7's, not this audit's.

## Flags per 1,000 graded words

The App arm ran the app's own CoreML decoder at 40 ms per inference; **strict** is Muraja's strict mode, **balanced** its shipped default. The denominator is the graded words (an ungraded word is stratum S5, not a pass); each interval resamples whole clips.

| scoring | graded words | flagged words | per 1,000 | 95% CI |
| --- | ---: | ---: | ---: | --- |
| strict | 48,593 | 5,157 | 106.1 | 101.4–110.9 |
| balanced (shipped default) | 48,659 | 4,325 | 88.9 | 84.6–93.2 |

### By cause

| cause | strict per 1,000 | 95% CI | balanced per 1,000 | 95% CI |
| --- | ---: | --- | ---: | --- |
| tashkeel | 14.4 | 13.2–15.6 | 14.7 | 13.5–16.0 |
| soft_pair | 9.8 | 8.8–10.8 | 2.7 | 2.2–3.1 |
| target_pair | 1.3 | 1.0–1.7 | 1.3 | 1.0–1.6 |
| other_consonant | 59.1 | 56.1–62.2 | 56.7 | 53.8–59.8 |
| shaddah | 6.2 | 5.4–6.9 | 2.4 | 1.9–2.9 |
| skipped | 9.1 | 7.6–10.9 | 8.6 | 7.1–10.2 |
| unrecorded | 10.0 | 9.1–10.9 | 6.1 | 5.4–6.8 |

## Oracle-explained share

The oracle's perfect decode flags 2 words of its own; the App also flags 1 of them: the oracle explains 0.019% of the App arm's 5,157 flagged words.

## P2: the CoreML-vs-PyTorch bridge

The bridge reuses the committed round-3 candidate EMA PyTorch decode `tlog.gating.r3_candidate_ema.jsonl.gz` — the same weights v1.0.7 ships, so no GPU run is spent re-deriving it — graded through the same session request as the App arm. Agreement is on the strict outcome (flagged / passed / ungraded), word by word.

Bar: 97%.

Seeded subset: 2,183 words, 2,150 agree (98.5%) — at or above the bar.
All clips (secondary): 57,935 words, 56,768 agree (98.0%).

Disagreeing words: 33 (the table below is exhaustive).
- the App arm flags what PyTorch passes: 15
- PyTorch flags what the App arm passes: 13 — the App arm's flag is below the PyTorch decode at these words.
- graded on one side only (ungraded against a grade): 11

| item | word | App arm | PyTorch |
| --- | ---: | --- | --- |
| 18_2_4002924421.flac#0 | 4 | flagged | passed |
| 1_7_5101853058.flac#0 | 9 | passed | ungraded |
| 1_7_7986943062.flac#0 | 9 | ungraded | flagged |
| 20_45_1560216144.flac#0 | 2 | flagged | ungraded |
| 20_45_1560216144.flac#0 | 3 | passed | ungraded |
| 20_45_1560216144.flac#0 | 4 | passed | ungraded |
| 20_45_1560216144.flac#0 | 5 | passed | ungraded |
| 20_45_1560216144.flac#0 | 6 | flagged | ungraded |
| 20_45_1560216144.flac#0 | 7 | flagged | ungraded |
| 29_45_163396162.flac#0 | 12 | passed | flagged |
| 2_127_9437504828.flac#0 | 4 | passed | flagged |
| 2_139_5106049286.flac#0 | 6 | passed | flagged |
| 2_236_4931181261.flac#0 | 18 | flagged | passed |
| 2_253_4703212957.flac#0 | 38 | passed | flagged |
| 2_3_8051759077.flac#0 | 1 | flagged | passed |
| 2_3_8051759077.flac#0 | 2 | flagged | passed |
| 2_3_8051759077.flac#0 | 4 | passed | flagged |
| 2_3_8051759077.flac#0 | 7 | passed | flagged |
| 2_92_2598961049.flac#0 | 6 | ungraded | flagged |
| 33_25_7700327261.flac#0 | 10 | flagged | passed |
| 33_25_7700327261.flac#0 | 15 | passed | flagged |
| 34_32_4194859777.flac#0 | 8 | flagged | passed |
| 35_8_8468571207.flac#0 | 10 | passed | flagged |
| 36_32_1920528665.flac#0 | 5 | flagged | passed |
| 47_19_5205807113.flac#0 | 11 | flagged | ungraded |
| 50_4_1514791089.flac#0 | 4 | passed | flagged |
| 53_9_5396664572.flac#0 | 5 | passed | flagged |
| 60_6_undefined_EZVJEKj.flac#0 | 5 | flagged | passed |
| 64_14_1876241576.flac#0 | 2 | flagged | passed |
| 67_5_2134960157.flac#0 | 4 | flagged | passed |
| 67_5_2134960157.flac#0 | 6 | passed | flagged |
| 76_1_2431673772.flac#0 | 1 | ungraded | passed |
| 76_1_2431673772.flac#0 | 7 | flagged | passed |

## P3: cadence

Device logs, previews per window:

| log | preview:window |
| --- | ---: |
| 69#1 | 5.26 |
| 70#1 | 5.22 |
| 70#2 | 5.05 |
| 71#1 | 4.83 |
| 71#2 | 5.06 |
| surah30_page405_shared_opening_correction | 5.30 |
| surah3_page51_ayah15_mutashabihat_wrong_words | 4.89 |
| surah3_page55_ayah42_mutashabihat_creep_jump | 5.56 |
| surah3_page58_far_reread_pause_residue | 5.79 |

The App arm's own cadence on its seeded subsets — previews per full window, pooled over the subset (flushes and the closing stop are neither). A clip that ran no window still counts its previews and is listed in its own column; a clip whose trace is empty — its session graded no word — has no word to pool and is listed beside the clips that do:

| traces | latency | clips with rows | previews | windows | preview:window | no window | no rows |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| p3_0 | 0 ms | 484 | 16,315 | 1,965 | 8.30 | 186 | 16 |
| p2 | 40 ms | 195 | 6,834 | 840 | 8.14 | 76 | 5 |
| p3_130 | 130 ms | 484 | 15,321 | 1,966 | 7.79 | 185 | 16 |

Device logs and the App arm read the same scheduler facet, but not under the same conditions: the logs are whole device sessions (their length sets how many previews fit), so a larger App ratio means these clips had more previews per window, not that the scheduler differs.

The App arm ran at 0 ms, 130 ms per inference on 500 clips (seed 309).

| scoring | words | changed | share |
| --- | ---: | ---: | ---: |
| strict | 5,006 | 59 | 1.2% |
| balanced (shipped default) | 5,006 | 54 | 1.1% |

No scoring changes more than 2% of word outcomes between the latency settings.

## Candidate strata

| stratum | words | catch-loss words |
| --- | ---: | ---: |
| S1: the App flags and the oracle passes | 5,151 | 0 |
| S1-pending: the App flags, no oracle verdict for the word | 5 | 0 |
| S3: the App passes and the base or two detectors flag | 1,459 | 0 |
| S4: nothing flags | 38,598 | 0 |
| S5: the App left the word ungraded | 9,342 | 0 |
| oracle-explained: the App flags and the oracle flags too | 1 | 0 |
| residual: the App passes and one non-base detector flags | 3,379 | 0 |

The App-flag pool by cause and base-detector agreement — strata S1 (the App flags and the oracle passes) and S1-pending (the App flags, no oracle verdict):

| cause | base detector flags | words |
| --- | --- | ---: |
| other_consonant | False | 1,175 |
| other_consonant | True | 1,698 |
| shaddah | False | 77 |
| shaddah | True | 223 |
| skipped | False | 147 |
| skipped | True | 296 |
| soft_pair | False | 130 |
| soft_pair | True | 346 |
| target_pair | False | 24 |
| target_pair | True | 41 |
| tashkeel | False | 283 |
| tashkeel | True | 416 |
| unrecorded | False | 98 |
| unrecorded | True | 386 |

Round 2's catch-loss worklist (A5b) marks 150 words; 0 of them are in this population. The worklist's words are look-tranche rows, so a confirm.1/confirm.2 population can match none.

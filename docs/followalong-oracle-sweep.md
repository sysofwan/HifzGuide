# Follow-along audit: the oracle and planted-mistake sweeps (#308)

Generated from `docs/followalong-oracle-sweep.json` by `python -m tadabur.followalong_audit oracle-sweep`; generated, do not edit it by hand.

The engine is fed a perfect decode: the reference phonemes placed on the clock by the Oracle arm, so no model and no ear is involved. Every word the engine does not grade correct is therefore a follow-along or reference fault, not a reciter's mistake.

Seed 308, Muraja `028bc513eeb233caa7240a5c37c423da59f3d9d7`, modes strict, balanced (shipped default).

## Pace

7.168 Characters per second — the median over the speech span of 5,575 aligned clips (a clip's first word start to its last word end). The mushaf sweep lays every ayah's words end to end at this pace.

## Clean sweep: the tlog clips

5,742 gating clips, each at its own phase A word times.
A word with no kept grade — no row in the engine's `final`, or only `pending` — is counted as not graded, never as a finding.

| mode | graded words | not graded | false flags per 1,000 words |
| --- | ---: | ---: | ---: |
| strict | 57,690 | 246 | 0.035 |
| balanced (shipped default) | 57,705 | 231 | 0.035 |

By cause and position (per 1,000 graded words):

| mode | error-kind rates | position rates |
| --- | --- | --- |
| strict | skipped 0.017, tashkeel 0.017 | first 0.000, last 0.178, middle 0.022 |
| balanced (shipped default) | skipped 0.017, tashkeel 0.017 | first 0.000, last 0.178, middle 0.022 |

## Clean sweep: the mushaf at the tlog pace

6,236 ayat; 6,228 graded and 8 the phonetizer cannot read (tallied, never dropped), at 7.168 Characters per second.

Unreadable ayat: 55:17 (KeyError: 'ن'), 90:8 (KeyError: 'ن'), 90:9 (KeyError: 'ن'), 90:10 (KeyError: 'ن'), 106:1 (KeyError: 'ش'), 106:2 (KeyError: 'ف'), 106:3 (KeyError: 'ت'), 106:4 (KeyError: 'ف').
A word with no kept grade — no row in the engine's `final`, or only `pending` — is counted as not graded, never as a finding.

| mode | graded words | not graded | false flags per 1,000 words |
| --- | ---: | ---: | ---: |
| strict | 76,610 | 794 | 0.026 |
| balanced (shipped default) | 76,622 | 782 | 0.026 |

By cause and position (per 1,000 graded words):

| mode | error-kind rates | position rates |
| --- | --- | --- |
| strict | tashkeel 0.026 | first 0.000, last 0.341, middle 0.000 |
| balanced (shipped default) | tashkeel 0.026 | first 0.000, last 0.340, middle 0.000 |

## Planted mistakes

One seeded mistake per kind per clip, planted before the reference is placed. **Caught** means the planted word's kept grade is below correct. A word with no kept grade — no row in the engine's `final`, or only `pending` — is `not_graded`: reported apart and counted as neither caught nor missed. Rates are caught over graded planted words.

| kind | mode | position | word length | graded | caught | not_graded | rate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| add_word | balanced (shipped default) | first | 1-2 | 83 | 0 | 2 | 0.000 |
| add_word | balanced (shipped default) | first | 11+ | 21 | 1 | 4 | 0.048 |
| add_word | balanced (shipped default) | first | 3-4 | 221 | 0 | 11 | 0.000 |
| add_word | balanced (shipped default) | first | 5-6 | 361 | 0 | 16 | 0.000 |
| add_word | balanced (shipped default) | first | 7-10 | 302 | 1 | 22 | 0.003 |
| add_word | balanced (shipped default) | last | 11+ | 50 | 0 | 1 | 0.000 |
| add_word | balanced (shipped default) | last | 3-4 | 49 | 0 | 3 | 0.000 |
| add_word | balanced (shipped default) | last | 5-6 | 268 | 0 | 43 | 0.000 |
| add_word | balanced (shipped default) | last | 7-10 | 466 | 0 | 49 | 0.000 |
| add_word | balanced (shipped default) | middle | 1-2 | 271 | 0 | 3 | 0.000 |
| add_word | balanced (shipped default) | middle | 11+ | 59 | 0 | 3 | 0.000 |
| add_word | balanced (shipped default) | middle | 3-4 | 1,149 | 24 | 34 | 0.021 |
| add_word | balanced (shipped default) | middle | 5-6 | 1,216 | 11 | 37 | 0.009 |
| add_word | balanced (shipped default) | middle | 7-10 | 962 | 3 | 36 | 0.003 |
| add_word | strict | first | 1-2 | 83 | 0 | 2 | 0.000 |
| add_word | strict | first | 11+ | 21 | 1 | 4 | 0.048 |
| add_word | strict | first | 3-4 | 221 | 0 | 11 | 0.000 |
| add_word | strict | first | 5-6 | 361 | 0 | 16 | 0.000 |
| add_word | strict | first | 7-10 | 302 | 1 | 22 | 0.003 |
| add_word | strict | last | 11+ | 50 | 0 | 1 | 0.000 |
| add_word | strict | last | 3-4 | 48 | 0 | 4 | 0.000 |
| add_word | strict | last | 5-6 | 266 | 0 | 45 | 0.000 |
| add_word | strict | last | 7-10 | 465 | 0 | 50 | 0.000 |
| add_word | strict | middle | 1-2 | 271 | 0 | 3 | 0.000 |
| add_word | strict | middle | 11+ | 59 | 0 | 3 | 0.000 |
| add_word | strict | middle | 3-4 | 1,149 | 24 | 34 | 0.021 |
| add_word | strict | middle | 5-6 | 1,216 | 11 | 37 | 0.009 |
| add_word | strict | middle | 7-10 | 962 | 3 | 36 | 0.003 |
| consonant_pair:ت↔ط | balanced (shipped default) | first | 11+ | 28 | 0 | 0 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | first | 3-4 | 46 | 0 | 0 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | first | 5-6 | 196 | 0 | 2 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | first | 7-10 | 178 | 0 | 4 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | last | 11+ | 46 | 0 | 1 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | last | 3-4 | 26 | 0 | 7 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | last | 5-6 | 114 | 0 | 11 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | last | 7-10 | 560 | 0 | 41 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | middle | 1-2 | 21 | 0 | 0 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | middle | 11+ | 124 | 0 | 0 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | middle | 3-4 | 337 | 0 | 4 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | middle | 5-6 | 1,146 | 19 | 4 | 0.017 |
| consonant_pair:ت↔ط | balanced (shipped default) | middle | 7-10 | 1,205 | 13 | 4 | 0.011 |
| consonant_pair:ت↔ط | strict | first | 11+ | 27 | 17 | 1 | 0.630 |
| consonant_pair:ت↔ط | strict | first | 3-4 | 46 | 45 | 0 | 0.978 |
| consonant_pair:ت↔ط | strict | first | 5-6 | 196 | 183 | 2 | 0.934 |
| consonant_pair:ت↔ط | strict | first | 7-10 | 178 | 149 | 4 | 0.837 |
| consonant_pair:ت↔ط | strict | last | 11+ | 46 | 43 | 1 | 0.935 |
| consonant_pair:ت↔ط | strict | last | 3-4 | 24 | 12 | 9 | 0.500 |
| consonant_pair:ت↔ط | strict | last | 5-6 | 112 | 68 | 13 | 0.607 |
| consonant_pair:ت↔ط | strict | last | 7-10 | 558 | 477 | 43 | 0.855 |
| consonant_pair:ت↔ط | strict | middle | 1-2 | 21 | 18 | 0 | 0.857 |
| consonant_pair:ت↔ط | strict | middle | 11+ | 122 | 69 | 2 | 0.566 |
| consonant_pair:ت↔ط | strict | middle | 3-4 | 324 | 190 | 17 | 0.586 |
| consonant_pair:ت↔ط | strict | middle | 5-6 | 1,111 | 682 | 39 | 0.614 |
| consonant_pair:ت↔ط | strict | middle | 7-10 | 1,194 | 679 | 15 | 0.569 |
| consonant_pair:ح↔ه | balanced (shipped default) | first | 1-2 | 11 | 0 | 0 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | first | 11+ | 16 | 0 | 0 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | first | 3-4 | 125 | 0 | 0 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | first | 5-6 | 138 | 1 | 3 | 0.007 |
| consonant_pair:ح↔ه | balanced (shipped default) | first | 7-10 | 198 | 0 | 5 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | last | 11+ | 9 | 0 | 0 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | last | 3-4 | 69 | 0 | 4 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | last | 5-6 | 163 | 0 | 24 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | last | 7-10 | 268 | 0 | 9 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | middle | 1-2 | 215 | 1 | 7 | 0.005 |
| consonant_pair:ح↔ه | balanced (shipped default) | middle | 11+ | 94 | 0 | 0 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | middle | 3-4 | 716 | 0 | 7 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | middle | 5-6 | 1,418 | 65 | 9 | 0.046 |
| consonant_pair:ح↔ه | balanced (shipped default) | middle | 7-10 | 976 | 22 | 4 | 0.023 |
| consonant_pair:ح↔ه | strict | first | 1-2 | 11 | 10 | 0 | 0.909 |
| consonant_pair:ح↔ه | strict | first | 11+ | 16 | 12 | 0 | 0.750 |
| consonant_pair:ح↔ه | strict | first | 3-4 | 125 | 124 | 0 | 0.992 |
| consonant_pair:ح↔ه | strict | first | 5-6 | 138 | 131 | 3 | 0.949 |
| consonant_pair:ح↔ه | strict | first | 7-10 | 197 | 163 | 6 | 0.827 |
| consonant_pair:ح↔ه | strict | last | 11+ | 9 | 2 | 0 | 0.222 |
| consonant_pair:ح↔ه | strict | last | 3-4 | 68 | 58 | 5 | 0.853 |
| consonant_pair:ح↔ه | strict | last | 5-6 | 153 | 100 | 34 | 0.654 |
| consonant_pair:ح↔ه | strict | last | 7-10 | 267 | 130 | 10 | 0.487 |
| consonant_pair:ح↔ه | strict | middle | 1-2 | 199 | 181 | 23 | 0.910 |
| consonant_pair:ح↔ه | strict | middle | 11+ | 94 | 42 | 0 | 0.447 |
| consonant_pair:ح↔ه | strict | middle | 3-4 | 692 | 489 | 31 | 0.707 |
| consonant_pair:ح↔ه | strict | middle | 5-6 | 1,386 | 751 | 41 | 0.542 |
| consonant_pair:ح↔ه | strict | middle | 7-10 | 963 | 567 | 17 | 0.589 |
| consonant_pair:ذ↔ز | balanced (shipped default) | first | 1-2 | 45 | 0 | 1 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | first | 11+ | 11 | 0 | 0 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | first | 3-4 | 102 | 0 | 1 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | first | 5-6 | 210 | 0 | 2 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | first | 7-10 | 197 | 0 | 1 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | last | 11+ | 8 | 0 | 0 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | last | 3-4 | 8 | 0 | 3 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | last | 5-6 | 71 | 0 | 1 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | last | 7-10 | 156 | 0 | 36 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | middle | 1-2 | 62 | 7 | 3 | 0.113 |
| consonant_pair:ذ↔ز | balanced (shipped default) | middle | 11+ | 21 | 0 | 0 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | middle | 3-4 | 284 | 2 | 2 | 0.007 |
| consonant_pair:ذ↔ز | balanced (shipped default) | middle | 5-6 | 1,249 | 0 | 6 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | middle | 7-10 | 514 | 0 | 1 | 0.000 |
| consonant_pair:ذ↔ز | strict | first | 1-2 | 45 | 45 | 1 | 1.000 |
| consonant_pair:ذ↔ز | strict | first | 11+ | 11 | 11 | 0 | 1.000 |
| consonant_pair:ذ↔ز | strict | first | 3-4 | 102 | 100 | 1 | 0.980 |
| consonant_pair:ذ↔ز | strict | first | 5-6 | 210 | 204 | 2 | 0.971 |
| consonant_pair:ذ↔ز | strict | first | 7-10 | 197 | 189 | 1 | 0.959 |
| consonant_pair:ذ↔ز | strict | last | 11+ | 8 | 2 | 0 | 0.250 |
| consonant_pair:ذ↔ز | strict | last | 3-4 | 7 | 6 | 4 | 0.857 |
| consonant_pair:ذ↔ز | strict | last | 5-6 | 70 | 60 | 2 | 0.857 |
| consonant_pair:ذ↔ز | strict | last | 7-10 | 155 | 116 | 37 | 0.748 |
| consonant_pair:ذ↔ز | strict | middle | 1-2 | 62 | 23 | 3 | 0.371 |
| consonant_pair:ذ↔ز | strict | middle | 11+ | 21 | 19 | 0 | 0.905 |
| consonant_pair:ذ↔ز | strict | middle | 3-4 | 272 | 215 | 14 | 0.790 |
| consonant_pair:ذ↔ز | strict | middle | 5-6 | 1,231 | 1,037 | 24 | 0.842 |
| consonant_pair:ذ↔ز | strict | middle | 7-10 | 512 | 452 | 3 | 0.883 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | first | 1-2 | 44 | 44 | 1 | 1.000 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | first | 11+ | 4 | 4 | 0 | 1.000 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | first | 3-4 | 118 | 117 | 1 | 0.992 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | first | 5-6 | 229 | 223 | 2 | 0.974 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | first | 7-10 | 203 | 192 | 1 | 0.946 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | last | 11+ | 6 | 2 | 0 | 0.333 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | last | 3-4 | 7 | 4 | 0 | 0.571 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | last | 5-6 | 76 | 54 | 8 | 0.711 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | last | 7-10 | 137 | 100 | 33 | 0.730 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | middle | 1-2 | 59 | 25 | 4 | 0.424 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | middle | 11+ | 22 | 20 | 0 | 0.909 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | middle | 3-4 | 264 | 192 | 11 | 0.727 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | middle | 5-6 | 1,074 | 843 | 21 | 0.785 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | middle | 7-10 | 418 | 341 | 4 | 0.816 |
| consonant_pair:ذ↔ظ | strict | first | 1-2 | 44 | 44 | 1 | 1.000 |
| consonant_pair:ذ↔ظ | strict | first | 11+ | 4 | 4 | 0 | 1.000 |
| consonant_pair:ذ↔ظ | strict | first | 3-4 | 118 | 117 | 1 | 0.992 |
| consonant_pair:ذ↔ظ | strict | first | 5-6 | 229 | 223 | 2 | 0.974 |
| consonant_pair:ذ↔ظ | strict | first | 7-10 | 203 | 194 | 1 | 0.956 |
| consonant_pair:ذ↔ظ | strict | last | 11+ | 6 | 2 | 0 | 0.333 |
| consonant_pair:ذ↔ظ | strict | last | 3-4 | 6 | 5 | 1 | 0.833 |
| consonant_pair:ذ↔ظ | strict | last | 5-6 | 75 | 62 | 9 | 0.827 |
| consonant_pair:ذ↔ظ | strict | last | 7-10 | 137 | 103 | 33 | 0.752 |
| consonant_pair:ذ↔ظ | strict | middle | 1-2 | 59 | 25 | 4 | 0.424 |
| consonant_pair:ذ↔ظ | strict | middle | 11+ | 22 | 21 | 0 | 0.955 |
| consonant_pair:ذ↔ظ | strict | middle | 3-4 | 264 | 194 | 11 | 0.735 |
| consonant_pair:ذ↔ظ | strict | middle | 5-6 | 1,074 | 931 | 21 | 0.867 |
| consonant_pair:ذ↔ظ | strict | middle | 7-10 | 418 | 351 | 4 | 0.840 |
| consonant_pair:س↔ش | balanced (shipped default) | first | 11+ | 9 | 8 | 0 | 0.889 |
| consonant_pair:س↔ش | balanced (shipped default) | first | 3-4 | 32 | 20 | 3 | 0.625 |
| consonant_pair:س↔ش | balanced (shipped default) | first | 5-6 | 76 | 57 | 1 | 0.750 |
| consonant_pair:س↔ش | balanced (shipped default) | first | 7-10 | 88 | 71 | 3 | 0.807 |
| consonant_pair:س↔ش | balanced (shipped default) | last | 11+ | 46 | 37 | 0 | 0.804 |
| consonant_pair:س↔ش | balanced (shipped default) | last | 3-4 | 30 | 30 | 4 | 1.000 |
| consonant_pair:س↔ش | balanced (shipped default) | last | 5-6 | 157 | 127 | 24 | 0.809 |
| consonant_pair:س↔ش | balanced (shipped default) | last | 7-10 | 610 | 446 | 21 | 0.731 |
| consonant_pair:س↔ش | balanced (shipped default) | middle | 1-2 | 16 | 16 | 2 | 1.000 |
| consonant_pair:س↔ش | balanced (shipped default) | middle | 11+ | 70 | 57 | 1 | 0.814 |
| consonant_pair:س↔ش | balanced (shipped default) | middle | 3-4 | 412 | 334 | 22 | 0.811 |
| consonant_pair:س↔ش | balanced (shipped default) | middle | 5-6 | 705 | 602 | 7 | 0.854 |
| consonant_pair:س↔ش | balanced (shipped default) | middle | 7-10 | 1,008 | 794 | 19 | 0.788 |
| consonant_pair:س↔ش | strict | first | 11+ | 9 | 8 | 0 | 0.889 |
| consonant_pair:س↔ش | strict | first | 3-4 | 32 | 20 | 3 | 0.625 |
| consonant_pair:س↔ش | strict | first | 5-6 | 76 | 57 | 1 | 0.750 |
| consonant_pair:س↔ش | strict | first | 7-10 | 88 | 72 | 3 | 0.818 |
| consonant_pair:س↔ش | strict | last | 11+ | 46 | 37 | 0 | 0.804 |
| consonant_pair:س↔ش | strict | last | 3-4 | 30 | 30 | 4 | 1.000 |
| consonant_pair:س↔ش | strict | last | 5-6 | 156 | 140 | 25 | 0.897 |
| consonant_pair:س↔ش | strict | last | 7-10 | 610 | 454 | 21 | 0.744 |
| consonant_pair:س↔ش | strict | middle | 1-2 | 16 | 16 | 2 | 1.000 |
| consonant_pair:س↔ش | strict | middle | 11+ | 70 | 61 | 1 | 0.871 |
| consonant_pair:س↔ش | strict | middle | 3-4 | 412 | 355 | 22 | 0.862 |
| consonant_pair:س↔ش | strict | middle | 5-6 | 705 | 635 | 7 | 0.901 |
| consonant_pair:س↔ش | strict | middle | 7-10 | 1,008 | 820 | 19 | 0.813 |
| consonant_pair:س↔ص | balanced (shipped default) | first | 11+ | 6 | 0 | 0 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | first | 3-4 | 45 | 0 | 2 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | first | 5-6 | 150 | 0 | 1 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | first | 7-10 | 105 | 0 | 6 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | last | 11+ | 49 | 0 | 0 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | last | 3-4 | 22 | 0 | 4 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | last | 5-6 | 173 | 1 | 15 | 0.006 |
| consonant_pair:س↔ص | balanced (shipped default) | last | 7-10 | 568 | 0 | 13 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | middle | 1-2 | 15 | 0 | 0 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | middle | 11+ | 62 | 0 | 0 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | middle | 3-4 | 357 | 0 | 7 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | middle | 5-6 | 714 | 0 | 3 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | middle | 7-10 | 1,115 | 0 | 4 | 0.000 |
| consonant_pair:س↔ص | strict | first | 11+ | 6 | 6 | 0 | 1.000 |
| consonant_pair:س↔ص | strict | first | 3-4 | 45 | 41 | 2 | 0.911 |
| consonant_pair:س↔ص | strict | first | 5-6 | 150 | 144 | 1 | 0.960 |
| consonant_pair:س↔ص | strict | first | 7-10 | 105 | 104 | 6 | 0.990 |
| consonant_pair:س↔ص | strict | last | 11+ | 49 | 42 | 0 | 0.857 |
| consonant_pair:س↔ص | strict | last | 3-4 | 22 | 22 | 4 | 1.000 |
| consonant_pair:س↔ص | strict | last | 5-6 | 165 | 147 | 23 | 0.891 |
| consonant_pair:س↔ص | strict | last | 7-10 | 564 | 422 | 17 | 0.748 |
| consonant_pair:س↔ص | strict | middle | 1-2 | 13 | 13 | 2 | 1.000 |
| consonant_pair:س↔ص | strict | middle | 11+ | 62 | 54 | 0 | 0.871 |
| consonant_pair:س↔ص | strict | middle | 3-4 | 344 | 270 | 20 | 0.785 |
| consonant_pair:س↔ص | strict | middle | 5-6 | 700 | 574 | 17 | 0.820 |
| consonant_pair:س↔ص | strict | middle | 7-10 | 1,097 | 886 | 22 | 0.808 |
| consonant_pair:ض↔ظ | balanced (shipped default) | first | 3-4 | 11 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | first | 5-6 | 15 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | first | 7-10 | 18 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | last | 11+ | 44 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | last | 3-4 | 1 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | last | 5-6 | 96 | 0 | 11 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | last | 7-10 | 133 | 0 | 8 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | middle | 1-2 | 3 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | middle | 11+ | 15 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | middle | 3-4 | 358 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | middle | 5-6 | 320 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | middle | 7-10 | 296 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | strict | first | 3-4 | 11 | 11 | 0 | 1.000 |
| consonant_pair:ض↔ظ | strict | first | 5-6 | 15 | 15 | 0 | 1.000 |
| consonant_pair:ض↔ظ | strict | first | 7-10 | 18 | 17 | 0 | 0.944 |
| consonant_pair:ض↔ظ | strict | last | 11+ | 44 | 26 | 0 | 0.591 |
| consonant_pair:ض↔ظ | strict | last | 3-4 | 1 | 1 | 0 | 1.000 |
| consonant_pair:ض↔ظ | strict | last | 5-6 | 89 | 76 | 18 | 0.854 |
| consonant_pair:ض↔ظ | strict | last | 7-10 | 132 | 85 | 9 | 0.644 |
| consonant_pair:ض↔ظ | strict | middle | 1-2 | 3 | 2 | 0 | 0.667 |
| consonant_pair:ض↔ظ | strict | middle | 11+ | 15 | 11 | 0 | 0.733 |
| consonant_pair:ض↔ظ | strict | middle | 3-4 | 350 | 170 | 8 | 0.486 |
| consonant_pair:ض↔ظ | strict | middle | 5-6 | 316 | 195 | 4 | 0.617 |
| consonant_pair:ض↔ظ | strict | middle | 7-10 | 294 | 250 | 2 | 0.850 |
| consonant_pair:ط↔ق | balanced (shipped default) | first | 1-2 | 104 | 1 | 15 | 0.010 |
| consonant_pair:ط↔ق | balanced (shipped default) | first | 11+ | 6 | 6 | 0 | 1.000 |
| consonant_pair:ط↔ق | balanced (shipped default) | first | 3-4 | 150 | 42 | 0 | 0.280 |
| consonant_pair:ط↔ق | balanced (shipped default) | first | 5-6 | 230 | 200 | 1 | 0.870 |
| consonant_pair:ط↔ق | balanced (shipped default) | first | 7-10 | 104 | 88 | 5 | 0.846 |
| consonant_pair:ط↔ق | balanced (shipped default) | last | 11+ | 21 | 5 | 0 | 0.238 |
| consonant_pair:ط↔ق | balanced (shipped default) | last | 3-4 | 38 | 22 | 7 | 0.579 |
| consonant_pair:ط↔ق | balanced (shipped default) | last | 5-6 | 144 | 84 | 22 | 0.583 |
| consonant_pair:ط↔ق | balanced (shipped default) | last | 7-10 | 365 | 161 | 11 | 0.441 |
| consonant_pair:ط↔ق | balanced (shipped default) | middle | 1-2 | 36 | 30 | 0 | 0.833 |
| consonant_pair:ط↔ق | balanced (shipped default) | middle | 11+ | 45 | 27 | 0 | 0.600 |
| consonant_pair:ط↔ق | balanced (shipped default) | middle | 3-4 | 467 | 340 | 17 | 0.728 |
| consonant_pair:ط↔ق | balanced (shipped default) | middle | 5-6 | 736 | 592 | 27 | 0.804 |
| consonant_pair:ط↔ق | balanced (shipped default) | middle | 7-10 | 687 | 589 | 7 | 0.857 |
| consonant_pair:ط↔ق | strict | first | 1-2 | 104 | 1 | 15 | 0.010 |
| consonant_pair:ط↔ق | strict | first | 11+ | 6 | 6 | 0 | 1.000 |
| consonant_pair:ط↔ق | strict | first | 3-4 | 150 | 47 | 0 | 0.313 |
| consonant_pair:ط↔ق | strict | first | 5-6 | 230 | 201 | 1 | 0.874 |
| consonant_pair:ط↔ق | strict | first | 7-10 | 104 | 88 | 5 | 0.846 |
| consonant_pair:ط↔ق | strict | last | 11+ | 21 | 11 | 0 | 0.524 |
| consonant_pair:ط↔ق | strict | last | 3-4 | 38 | 23 | 7 | 0.605 |
| consonant_pair:ط↔ق | strict | last | 5-6 | 142 | 105 | 24 | 0.739 |
| consonant_pair:ط↔ق | strict | last | 7-10 | 365 | 173 | 11 | 0.474 |
| consonant_pair:ط↔ق | strict | middle | 1-2 | 36 | 30 | 0 | 0.833 |
| consonant_pair:ط↔ق | strict | middle | 11+ | 45 | 30 | 0 | 0.667 |
| consonant_pair:ط↔ق | strict | middle | 3-4 | 467 | 347 | 17 | 0.743 |
| consonant_pair:ط↔ق | strict | middle | 5-6 | 736 | 618 | 27 | 0.840 |
| consonant_pair:ط↔ق | strict | middle | 7-10 | 687 | 609 | 7 | 0.886 |
| consonant_pair:ق↔ك | balanced (shipped default) | first | 1-2 | 82 | 2 | 4 | 0.024 |
| consonant_pair:ق↔ك | balanced (shipped default) | first | 11+ | 10 | 0 | 0 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | first | 3-4 | 117 | 1 | 0 | 0.009 |
| consonant_pair:ق↔ك | balanced (shipped default) | first | 5-6 | 288 | 5 | 4 | 0.017 |
| consonant_pair:ق↔ك | balanced (shipped default) | first | 7-10 | 190 | 1 | 5 | 0.005 |
| consonant_pair:ق↔ك | balanced (shipped default) | last | 11+ | 15 | 0 | 0 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | last | 3-4 | 27 | 0 | 8 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | last | 5-6 | 167 | 0 | 15 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | last | 7-10 | 415 | 0 | 29 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | middle | 1-2 | 99 | 1 | 1 | 0.010 |
| consonant_pair:ق↔ك | balanced (shipped default) | middle | 11+ | 51 | 0 | 1 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | middle | 3-4 | 813 | 0 | 9 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | middle | 5-6 | 1,172 | 37 | 6 | 0.032 |
| consonant_pair:ق↔ك | balanced (shipped default) | middle | 7-10 | 915 | 13 | 8 | 0.014 |
| consonant_pair:ق↔ك | strict | first | 1-2 | 82 | 81 | 4 | 0.988 |
| consonant_pair:ق↔ك | strict | first | 11+ | 10 | 7 | 0 | 0.700 |
| consonant_pair:ق↔ك | strict | first | 3-4 | 117 | 114 | 0 | 0.974 |
| consonant_pair:ق↔ك | strict | first | 5-6 | 288 | 268 | 4 | 0.931 |
| consonant_pair:ق↔ك | strict | first | 7-10 | 190 | 182 | 5 | 0.958 |
| consonant_pair:ق↔ك | strict | last | 11+ | 15 | 6 | 0 | 0.400 |
| consonant_pair:ق↔ك | strict | last | 3-4 | 26 | 11 | 9 | 0.423 |
| consonant_pair:ق↔ك | strict | last | 5-6 | 164 | 122 | 18 | 0.744 |
| consonant_pair:ق↔ك | strict | last | 7-10 | 413 | 241 | 31 | 0.584 |
| consonant_pair:ق↔ك | strict | middle | 1-2 | 95 | 55 | 5 | 0.579 |
| consonant_pair:ق↔ك | strict | middle | 11+ | 51 | 34 | 1 | 0.667 |
| consonant_pair:ق↔ك | strict | middle | 3-4 | 785 | 571 | 37 | 0.727 |
| consonant_pair:ق↔ك | strict | middle | 5-6 | 1,143 | 934 | 35 | 0.817 |
| consonant_pair:ق↔ك | strict | middle | 7-10 | 882 | 646 | 41 | 0.732 |
| consonant_random | balanced (shipped default) | first | 1-2 | 58 | 37 | 4 | 0.638 |
| consonant_random | balanced (shipped default) | first | 11+ | 22 | 17 | 0 | 0.773 |
| consonant_random | balanced (shipped default) | first | 3-4 | 211 | 142 | 8 | 0.673 |
| consonant_random | balanced (shipped default) | first | 5-6 | 371 | 250 | 12 | 0.674 |
| consonant_random | balanced (shipped default) | first | 7-10 | 278 | 190 | 12 | 0.683 |
| consonant_random | balanced (shipped default) | last | 11+ | 38 | 15 | 1 | 0.395 |
| consonant_random | balanced (shipped default) | last | 3-4 | 61 | 37 | 7 | 0.607 |
| consonant_random | balanced (shipped default) | last | 5-6 | 299 | 157 | 46 | 0.525 |
| consonant_random | balanced (shipped default) | last | 7-10 | 507 | 273 | 34 | 0.538 |
| consonant_random | balanced (shipped default) | middle | 1-2 | 230 | 148 | 25 | 0.643 |
| consonant_random | balanced (shipped default) | middle | 11+ | 64 | 33 | 1 | 0.516 |
| consonant_random | balanced (shipped default) | middle | 3-4 | 1,067 | 776 | 91 | 0.727 |
| consonant_random | balanced (shipped default) | middle | 5-6 | 1,291 | 865 | 53 | 0.670 |
| consonant_random | balanced (shipped default) | middle | 7-10 | 918 | 578 | 33 | 0.630 |
| consonant_random | strict | first | 1-2 | 58 | 37 | 4 | 0.638 |
| consonant_random | strict | first | 11+ | 22 | 18 | 0 | 0.818 |
| consonant_random | strict | first | 3-4 | 211 | 170 | 8 | 0.806 |
| consonant_random | strict | first | 5-6 | 371 | 254 | 12 | 0.685 |
| consonant_random | strict | first | 7-10 | 278 | 194 | 12 | 0.698 |
| consonant_random | strict | last | 11+ | 38 | 15 | 1 | 0.395 |
| consonant_random | strict | last | 3-4 | 61 | 41 | 7 | 0.672 |
| consonant_random | strict | last | 5-6 | 298 | 183 | 47 | 0.614 |
| consonant_random | strict | last | 7-10 | 505 | 291 | 36 | 0.576 |
| consonant_random | strict | middle | 1-2 | 230 | 148 | 25 | 0.643 |
| consonant_random | strict | middle | 11+ | 64 | 37 | 1 | 0.578 |
| consonant_random | strict | middle | 3-4 | 1,066 | 806 | 92 | 0.756 |
| consonant_random | strict | middle | 5-6 | 1,290 | 898 | 54 | 0.696 |
| consonant_random | strict | middle | 7-10 | 918 | 628 | 33 | 0.684 |
| haraka_drop | balanced (shipped default) | first | 1-2 | 61 | 40 | 0 | 0.656 |
| haraka_drop | balanced (shipped default) | first | 11+ | 18 | 10 | 0 | 0.556 |
| haraka_drop | balanced (shipped default) | first | 3-4 | 210 | 138 | 3 | 0.657 |
| haraka_drop | balanced (shipped default) | first | 5-6 | 378 | 201 | 7 | 0.532 |
| haraka_drop | balanced (shipped default) | first | 7-10 | 296 | 177 | 5 | 0.598 |
| haraka_drop | balanced (shipped default) | last | 11+ | 68 | 29 | 0 | 0.426 |
| haraka_drop | balanced (shipped default) | last | 3-4 | 61 | 42 | 1 | 0.689 |
| haraka_drop | balanced (shipped default) | last | 5-6 | 312 | 175 | 17 | 0.561 |
| haraka_drop | balanced (shipped default) | last | 7-10 | 541 | 251 | 15 | 0.464 |
| haraka_drop | balanced (shipped default) | middle | 1-2 | 270 | 111 | 11 | 0.411 |
| haraka_drop | balanced (shipped default) | middle | 11+ | 59 | 29 | 0 | 0.492 |
| haraka_drop | balanced (shipped default) | middle | 3-4 | 1,137 | 625 | 60 | 0.550 |
| haraka_drop | balanced (shipped default) | middle | 5-6 | 1,249 | 682 | 47 | 0.546 |
| haraka_drop | balanced (shipped default) | middle | 7-10 | 888 | 439 | 28 | 0.494 |
| haraka_drop | strict | first | 1-2 | 61 | 40 | 0 | 0.656 |
| haraka_drop | strict | first | 11+ | 18 | 11 | 0 | 0.611 |
| haraka_drop | strict | first | 3-4 | 210 | 141 | 3 | 0.671 |
| haraka_drop | strict | first | 5-6 | 378 | 251 | 7 | 0.664 |
| haraka_drop | strict | first | 7-10 | 296 | 201 | 5 | 0.679 |
| haraka_drop | strict | last | 11+ | 68 | 35 | 0 | 0.515 |
| haraka_drop | strict | last | 3-4 | 59 | 43 | 3 | 0.729 |
| haraka_drop | strict | last | 5-6 | 311 | 195 | 18 | 0.627 |
| haraka_drop | strict | last | 7-10 | 539 | 290 | 17 | 0.538 |
| haraka_drop | strict | middle | 1-2 | 270 | 111 | 11 | 0.411 |
| haraka_drop | strict | middle | 11+ | 59 | 32 | 0 | 0.542 |
| haraka_drop | strict | middle | 3-4 | 1,131 | 660 | 66 | 0.584 |
| haraka_drop | strict | middle | 5-6 | 1,246 | 753 | 50 | 0.604 |
| haraka_drop | strict | middle | 7-10 | 887 | 511 | 29 | 0.576 |
| haraka_swap | balanced (shipped default) | first | 1-2 | 67 | 67 | 4 | 1.000 |
| haraka_swap | balanced (shipped default) | first | 11+ | 18 | 17 | 1 | 0.944 |
| haraka_swap | balanced (shipped default) | first | 3-4 | 220 | 208 | 1 | 0.945 |
| haraka_swap | balanced (shipped default) | first | 5-6 | 378 | 351 | 10 | 0.929 |
| haraka_swap | balanced (shipped default) | first | 7-10 | 279 | 250 | 10 | 0.896 |
| haraka_swap | balanced (shipped default) | last | 11+ | 43 | 26 | 0 | 0.605 |
| haraka_swap | balanced (shipped default) | last | 3-4 | 60 | 52 | 2 | 0.867 |
| haraka_swap | balanced (shipped default) | last | 5-6 | 307 | 201 | 17 | 0.655 |
| haraka_swap | balanced (shipped default) | last | 7-10 | 527 | 321 | 14 | 0.609 |
| haraka_swap | balanced (shipped default) | middle | 1-2 | 255 | 159 | 21 | 0.624 |
| haraka_swap | balanced (shipped default) | middle | 11+ | 55 | 33 | 1 | 0.600 |
| haraka_swap | balanced (shipped default) | middle | 3-4 | 1,102 | 763 | 73 | 0.692 |
| haraka_swap | balanced (shipped default) | middle | 5-6 | 1,277 | 933 | 42 | 0.731 |
| haraka_swap | balanced (shipped default) | middle | 7-10 | 930 | 650 | 28 | 0.699 |
| haraka_swap | strict | first | 1-2 | 67 | 67 | 4 | 1.000 |
| haraka_swap | strict | first | 11+ | 18 | 17 | 1 | 0.944 |
| haraka_swap | strict | first | 3-4 | 220 | 210 | 1 | 0.955 |
| haraka_swap | strict | first | 5-6 | 378 | 351 | 10 | 0.929 |
| haraka_swap | strict | first | 7-10 | 279 | 254 | 10 | 0.910 |
| haraka_swap | strict | last | 11+ | 43 | 26 | 0 | 0.605 |
| haraka_swap | strict | last | 3-4 | 59 | 54 | 3 | 0.915 |
| haraka_swap | strict | last | 5-6 | 306 | 235 | 18 | 0.768 |
| haraka_swap | strict | last | 7-10 | 527 | 344 | 14 | 0.653 |
| haraka_swap | strict | middle | 1-2 | 255 | 159 | 21 | 0.624 |
| haraka_swap | strict | middle | 11+ | 55 | 36 | 1 | 0.655 |
| haraka_swap | strict | middle | 3-4 | 1,102 | 776 | 73 | 0.704 |
| haraka_swap | strict | middle | 5-6 | 1,277 | 971 | 42 | 0.760 |
| haraka_swap | strict | middle | 7-10 | 930 | 700 | 28 | 0.753 |
| shaddah_drop | balanced (shipped default) | first | 1-2 | 17 | 0 | 0 | 0.000 |
| shaddah_drop | balanced (shipped default) | first | 11+ | 22 | 0 | 0 | 0.000 |
| shaddah_drop | balanced (shipped default) | first | 3-4 | 261 | 14 | 7 | 0.054 |
| shaddah_drop | balanced (shipped default) | first | 5-6 | 328 | 1 | 2 | 0.003 |
| shaddah_drop | balanced (shipped default) | first | 7-10 | 237 | 1 | 6 | 0.004 |
| shaddah_drop | balanced (shipped default) | last | 11+ | 55 | 0 | 0 | 0.000 |
| shaddah_drop | balanced (shipped default) | last | 3-4 | 91 | 0 | 16 | 0.000 |
| shaddah_drop | balanced (shipped default) | last | 5-6 | 157 | 4 | 19 | 0.025 |
| shaddah_drop | balanced (shipped default) | last | 7-10 | 292 | 1 | 14 | 0.003 |
| shaddah_drop | balanced (shipped default) | middle | 1-2 | 270 | 1 | 2 | 0.004 |
| shaddah_drop | balanced (shipped default) | middle | 11+ | 41 | 0 | 0 | 0.000 |
| shaddah_drop | balanced (shipped default) | middle | 3-4 | 1,408 | 3 | 5 | 0.002 |
| shaddah_drop | balanced (shipped default) | middle | 5-6 | 1,288 | 1 | 8 | 0.001 |
| shaddah_drop | balanced (shipped default) | middle | 7-10 | 621 | 4 | 0 | 0.006 |
| shaddah_drop | strict | first | 1-2 | 17 | 15 | 0 | 0.882 |
| shaddah_drop | strict | first | 11+ | 22 | 5 | 0 | 0.227 |
| shaddah_drop | strict | first | 3-4 | 261 | 102 | 7 | 0.391 |
| shaddah_drop | strict | first | 5-6 | 328 | 234 | 2 | 0.713 |
| shaddah_drop | strict | first | 7-10 | 235 | 110 | 8 | 0.468 |
| shaddah_drop | strict | last | 11+ | 55 | 0 | 0 | 0.000 |
| shaddah_drop | strict | last | 3-4 | 91 | 25 | 16 | 0.275 |
| shaddah_drop | strict | last | 5-6 | 157 | 62 | 19 | 0.395 |
| shaddah_drop | strict | last | 7-10 | 290 | 103 | 16 | 0.355 |
| shaddah_drop | strict | middle | 1-2 | 259 | 166 | 13 | 0.641 |
| shaddah_drop | strict | middle | 11+ | 41 | 16 | 0 | 0.390 |
| shaddah_drop | strict | middle | 3-4 | 1,383 | 436 | 30 | 0.315 |
| shaddah_drop | strict | middle | 5-6 | 1,263 | 449 | 33 | 0.356 |
| shaddah_drop | strict | middle | 7-10 | 618 | 157 | 3 | 0.254 |
| skip_word | balanced (shipped default) | first | 1-2 | 0 | 0 | 70 | — |
| skip_word | balanced (shipped default) | first | 11+ | 0 | 0 | 15 | — |
| skip_word | balanced (shipped default) | first | 3-4 | 0 | 0 | 240 | — |
| skip_word | balanced (shipped default) | first | 5-6 | 0 | 0 | 375 | — |
| skip_word | balanced (shipped default) | first | 7-10 | 1 | 1 | 280 | 1.000 |
| skip_word | balanced (shipped default) | last | 11+ | 0 | 0 | 46 | — |
| skip_word | balanced (shipped default) | last | 3-4 | 0 | 0 | 52 | — |
| skip_word | balanced (shipped default) | last | 5-6 | 0 | 0 | 309 | — |
| skip_word | balanced (shipped default) | last | 7-10 | 0 | 0 | 535 | — |
| skip_word | balanced (shipped default) | middle | 1-2 | 225 | 220 | 65 | 0.978 |
| skip_word | balanced (shipped default) | middle | 11+ | 61 | 61 | 13 | 1.000 |
| skip_word | balanced (shipped default) | middle | 3-4 | 820 | 819 | 333 | 0.999 |
| skip_word | balanced (shipped default) | middle | 5-6 | 1,042 | 1,041 | 289 | 0.999 |
| skip_word | balanced (shipped default) | middle | 7-10 | 740 | 739 | 227 | 0.999 |
| skip_word | strict | first | 1-2 | 0 | 0 | 70 | — |
| skip_word | strict | first | 11+ | 0 | 0 | 15 | — |
| skip_word | strict | first | 3-4 | 0 | 0 | 240 | — |
| skip_word | strict | first | 5-6 | 0 | 0 | 375 | — |
| skip_word | strict | first | 7-10 | 1 | 1 | 280 | 1.000 |
| skip_word | strict | last | 11+ | 0 | 0 | 46 | — |
| skip_word | strict | last | 3-4 | 0 | 0 | 52 | — |
| skip_word | strict | last | 5-6 | 0 | 0 | 309 | — |
| skip_word | strict | last | 7-10 | 0 | 0 | 535 | — |
| skip_word | strict | middle | 1-2 | 225 | 220 | 65 | 0.978 |
| skip_word | strict | middle | 11+ | 61 | 61 | 13 | 1.000 |
| skip_word | strict | middle | 3-4 | 820 | 819 | 333 | 0.999 |
| skip_word | strict | middle | 5-6 | 1,042 | 1,041 | 289 | 0.999 |
| skip_word | strict | middle | 7-10 | 740 | 739 | 227 | 0.999 |

Kinds no clip could carry, with the number of clips that could not: consonant_pair:ت↔ط (1,637), consonant_pair:ح↔ه (1,254), consonant_pair:ذ↔ز (2,747), consonant_pair:ذ↔ظ (2,995), consonant_pair:س↔ش (2,376), consonant_pair:س↔ص (2,306), consonant_pair:ض↔ظ (4,413), consonant_pair:ط↔ق (2,497), consonant_pair:ق↔ك (1,291), shaddah_drop (575), skip_word (4).

### Haraka drops: protected exemptions

A drop on و ا ء ي is labelled `protected`; every other letter is `other`.

| mode | protected | graded | caught | not_graded | rate |
| --- | --- | ---: | ---: | ---: | ---: |
| balanced (shipped default) | False | 4,499 | 2,948 | 181 | 0.655 |
| balanced (shipped default) | True | 1,049 | 1 | 13 | 0.001 |
| strict | False | 4,487 | 3,259 | 193 | 0.726 |
| strict | True | 1,046 | 15 | 16 | 0.014 |

Further letters the engine never flagged a drop on (observed, not assumed exempt): none.

### Added words: copied word vs the word after it

An added word is a copy of its neighbour inserted right after it. The harness reports the ayah's own phoneme words, so the copy has no word number of its own: the insertion can displace the grade of the copied word and of the word after it, and caught means a non-correct grade on either; the split is reported apart.

| mode | graded | caught | on the copied word | on the word after | not_graded | rate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| balanced (shipped default) | 5,478 | 40 | 27 | 13 | 264 | 0.007 |
| strict | 5,474 | 40 | 27 | 13 | 268 | 0.007 |

## Patterns

The clean sweeps' findings — every word the engine did not grade correct under a perfect decode — are pooled by cause (every grouping) and by reference form, where only the forms covering ten or more words are listed: a form seen once is not a pattern. A planted mistake the engine misses is catch capacity, not a finding, so it never enters a pattern. A pattern covering ten or more distinct ayah words is a candidate Muraja issue. The drafts are in the run note (`tadabur/followalong_audit/oracle_sweep/run_note.md`) and are not filed by this repo.

| pattern | words | findings | candidate |
| --- | ---: | ---: | --- |
| error_kind: tashkeel | 3 | 6 | False |
| error_kind: skipped | 1 | 2 | False |
| error_kind: tashkeel, position: last | 3 | 6 | False |
| error_kind: skipped, position: middle | 1 | 2 | False |
| boundary: False, error_kind: tashkeel | 2 | 4 | False |
| boundary: None, error_kind: tashkeel | 1 | 2 | False |
| boundary: True, error_kind: skipped | 1 | 2 | False |

### error_kind: tashkeel — 3 words

- 14:21 word 34 (balanced (shipped default)): tashkeelError, tashkeel, last, boundary None; form `ممممَحِۦۦص`
- 54:22 word 7 (balanced (shipped default)): tashkeelError, tashkeel, last, boundary False; form `ممممُددَكِر`
- 54:40 word 7 (balanced (shipped default)): tashkeelError, tashkeel, last, boundary False; form `ممممُددَكِر`

Replay evidence (first example, per-check statuses):

```json
[null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, {"errors": [{"group_index": 0, "kind": {"expected": "َ", "heard": null, "type": "tashkeel"}}], "quality": "pending", "score": 1, "withheld": "tashkeelError", "word": 34}, {"errors": [{"group_index": 0, "kind": {"expected": "َ", "heard": null, "type": "tashkeel"}}], "quality": "pending", "score": 1, "withheld": "tashkeelError", "word": 34}, {"errors": [{"group_index": 0, "kind": {"expected": "َ", "heard": null, "type": "tashkeel"}}], "quality": "pending", "score": 1, "withheld": "tashkeelError", "word": 34}]
```

### error_kind: skipped — 1 words

- 3:45 word 3 (balanced (shipped default)): skipped, skipped, middle, boundary True; form `لمَلَااااءِكَتُ`

Replay evidence (first example, per-check statuses):

```json
[null, null, null, null, null, null, null, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.9, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, null, null, null, null, null, null, null, null, null, null, null, null, null, null, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null]
```

### error_kind: tashkeel, position: last — 3 words

- 14:21 word 34 (balanced (shipped default)): tashkeelError, tashkeel, last, boundary None; form `ممممَحِۦۦص`
- 54:22 word 7 (balanced (shipped default)): tashkeelError, tashkeel, last, boundary False; form `ممممُددَكِر`
- 54:40 word 7 (balanced (shipped default)): tashkeelError, tashkeel, last, boundary False; form `ممممُددَكِر`

Replay evidence (first example, per-check statuses):

```json
[null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, {"errors": [{"group_index": 0, "kind": {"expected": "َ", "heard": null, "type": "tashkeel"}}], "quality": "pending", "score": 1, "withheld": "tashkeelError", "word": 34}, {"errors": [{"group_index": 0, "kind": {"expected": "َ", "heard": null, "type": "tashkeel"}}], "quality": "pending", "score": 1, "withheld": "tashkeelError", "word": 34}, {"errors": [{"group_index": 0, "kind": {"expected": "َ", "heard": null, "type": "tashkeel"}}], "quality": "pending", "score": 1, "withheld": "tashkeelError", "word": 34}]
```

### error_kind: skipped, position: middle — 1 words

- 3:45 word 3 (balanced (shipped default)): skipped, skipped, middle, boundary True; form `لمَلَااااءِكَتُ`

Replay evidence (first example, per-check statuses):

```json
[null, null, null, null, null, null, null, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.9, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, null, null, null, null, null, null, null, null, null, null, null, null, null, null, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null]
```

### boundary: False, error_kind: tashkeel — 2 words

- 54:22 word 7 (balanced (shipped default)): tashkeelError, tashkeel, last, boundary False; form `ممممُددَكِر`
- 54:40 word 7 (balanced (shipped default)): tashkeelError, tashkeel, last, boundary False; form `ممممُددَكِر`

Replay evidence (first example, per-check statuses):

```json
[null, null, null, null, null, null, null, null, null, {"errors": [{"group_index": 0, "kind": {"expected": "ُ", "heard": null, "type": "tashkeel"}}], "quality": "pending", "score": 1, "withheld": "tashkeelError", "word": 7}, {"errors": [{"group_index": 0, "kind": {"expected": "ُ", "heard": null, "type": "tashkeel"}}], "quality": "pending", "score": 1, "withheld": "tashkeelError", "word": 7}, null, null, null, null]
```

### boundary: None, error_kind: tashkeel — 1 words

- 14:21 word 34 (balanced (shipped default)): tashkeelError, tashkeel, last, boundary None; form `ممممَحِۦۦص`

Replay evidence (first example, per-check statuses):

```json
[null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, {"errors": [{"group_index": 0, "kind": {"expected": "َ", "heard": null, "type": "tashkeel"}}], "quality": "pending", "score": 1, "withheld": "tashkeelError", "word": 34}, {"errors": [{"group_index": 0, "kind": {"expected": "َ", "heard": null, "type": "tashkeel"}}], "quality": "pending", "score": 1, "withheld": "tashkeelError", "word": 34}, {"errors": [{"group_index": 0, "kind": {"expected": "َ", "heard": null, "type": "tashkeel"}}], "quality": "pending", "score": 1, "withheld": "tashkeelError", "word": 34}]
```

### boundary: True, error_kind: skipped — 1 words

- 3:45 word 3 (balanced (shipped default)): skipped, skipped, middle, boundary True; form `لمَلَااااءِكَتُ`

Replay evidence (first example, per-check statuses):

```json
[null, null, null, null, null, null, null, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.9, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, null, null, null, null, null, null, null, null, null, null, null, null, null, null, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 1, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, {"errors": [], "quality": "correct", "score": 0.7, "word": 3}, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null, null]
```

# Follow-along audit: the oracle and planted-mistake sweeps (#308)

Generated from `docs/followalong-oracle-sweep.json` by `python -m tadabur.followalong_audit oracle-sweep`; generated, do not edit it by hand.

The engine is fed a perfect decode: the reference phonemes placed on the clock by the Oracle arm, so no model and no ear is involved. Every word the engine does not grade correct is therefore a follow-along or reference fault, not a reciter's mistake.

Seed 308, Muraja `028bc513eeb233caa7240a5c37c423da59f3d9d7`, modes strict, balanced (shipped default).

## Pace

7.168 Characters per second — the median over the speech span of 5,575 aligned clips (a clip's first word start to its last word end). The mushaf sweep lays every ayah's words end to end at this pace.

## Clean sweep: the tlog clips

5,742 gating clips, each at its own phase A word times.
A word with no kept grade — no row in the engine's `final`, or only `pending` — is counted as not graded, never as a finding. Words, denominators and positions are Muraja's own phoneme words; an internal printed pause sign marks its word `waqf`, and a wasl merge recites two canonical words as one of them.

| mode | graded words | not graded | false flags per 1,000 words |
| --- | ---: | ---: | ---: |
| strict | 57,690 | 245 | 0.035 |
| balanced (shipped default) | 57,705 | 230 | 0.035 |

By cause and position (per 1,000 graded words):

| mode | error-kind rates | position rates |
| --- | --- | --- |
| strict | skipped 0.017, tashkeel 0.017 | first 0.000, last 0.178, middle 0.023, waqf 0.000 |
| balanced (shipped default) | skipped 0.017, tashkeel 0.017 | first 0.000, last 0.178, middle 0.023, waqf 0.000 |

## Clean sweep: the mushaf at the tlog pace

6,236 ayat; 6,228 graded and 8 the phonetizer cannot read (tallied, never dropped), at 7.168 Characters per second.

Unreadable ayat: 55:17 (KeyError: 'ن'), 90:8 (KeyError: 'ن'), 90:9 (KeyError: 'ن'), 90:10 (KeyError: 'ن'), 106:1 (KeyError: 'ش'), 106:2 (KeyError: 'ف'), 106:3 (KeyError: 'ت'), 106:4 (KeyError: 'ف').
A word with no kept grade — no row in the engine's `final`, or only `pending` — is counted as not graded, never as a finding. Words, denominators and positions are Muraja's own phoneme words; an internal printed pause sign marks its word `waqf`, and a wasl merge recites two canonical words as one of them.

| mode | graded words | not graded | false flags per 1,000 words |
| --- | ---: | ---: | ---: |
| strict | 76,610 | 788 | 0.026 |
| balanced (shipped default) | 76,622 | 776 | 0.026 |

By cause and position (per 1,000 graded words):

| mode | error-kind rates | position rates |
| --- | --- | --- |
| strict | tashkeel 0.026 | first 0.000, last 0.341, middle 0.000, waqf 0.000 |
| balanced (shipped default) | tashkeel 0.026 | first 0.000, last 0.340, middle 0.000, waqf 0.000 |

## Planted mistakes

One seeded mistake per kind per clip, planted in a canonical word before the reference is placed. **Caught** means the kept grade of the engine word the plant is recited in is below correct: a wasl merge recites two canonical words in one engine word, and an added word's copy has no engine word of its own. A word with no kept grade — no row in the engine's `final`, or only `pending` — is `not_graded`: reported apart and counted as neither caught nor missed. Rates are caught over graded planted words.

| kind | mode | position | word length | graded | caught | not_graded | rate |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| add_word | balanced (shipped default) | first | 1-2 | 83 | 0 | 2 | 0.000 |
| add_word | balanced (shipped default) | first | 11+ | 19 | 1 | 4 | 0.053 |
| add_word | balanced (shipped default) | first | 3-4 | 221 | 0 | 11 | 0.000 |
| add_word | balanced (shipped default) | first | 5-6 | 360 | 0 | 16 | 0.000 |
| add_word | balanced (shipped default) | first | 7-10 | 298 | 1 | 22 | 0.003 |
| add_word | balanced (shipped default) | last | 11+ | 50 | 0 | 1 | 0.000 |
| add_word | balanced (shipped default) | last | 3-4 | 49 | 0 | 3 | 0.000 |
| add_word | balanced (shipped default) | last | 5-6 | 268 | 0 | 43 | 0.000 |
| add_word | balanced (shipped default) | last | 7-10 | 466 | 0 | 49 | 0.000 |
| add_word | balanced (shipped default) | middle | 1-2 | 271 | 0 | 3 | 0.000 |
| add_word | balanced (shipped default) | middle | 11+ | 50 | 0 | 3 | 0.000 |
| add_word | balanced (shipped default) | middle | 3-4 | 1,128 | 24 | 34 | 0.021 |
| add_word | balanced (shipped default) | middle | 5-6 | 1,163 | 11 | 37 | 0.009 |
| add_word | balanced (shipped default) | middle | 7-10 | 892 | 3 | 36 | 0.003 |
| add_word | balanced (shipped default) | waqf | 11+ | 11 | 0 | 0 | 0.000 |
| add_word | balanced (shipped default) | waqf | 3-4 | 21 | 0 | 0 | 0.000 |
| add_word | balanced (shipped default) | waqf | 5-6 | 54 | 0 | 0 | 0.000 |
| add_word | balanced (shipped default) | waqf | 7-10 | 74 | 0 | 0 | 0.000 |
| add_word | strict | first | 1-2 | 83 | 0 | 2 | 0.000 |
| add_word | strict | first | 11+ | 19 | 1 | 4 | 0.053 |
| add_word | strict | first | 3-4 | 221 | 0 | 11 | 0.000 |
| add_word | strict | first | 5-6 | 360 | 0 | 16 | 0.000 |
| add_word | strict | first | 7-10 | 298 | 1 | 22 | 0.003 |
| add_word | strict | last | 11+ | 50 | 0 | 1 | 0.000 |
| add_word | strict | last | 3-4 | 48 | 0 | 4 | 0.000 |
| add_word | strict | last | 5-6 | 266 | 0 | 45 | 0.000 |
| add_word | strict | last | 7-10 | 465 | 0 | 50 | 0.000 |
| add_word | strict | middle | 1-2 | 271 | 0 | 3 | 0.000 |
| add_word | strict | middle | 11+ | 50 | 0 | 3 | 0.000 |
| add_word | strict | middle | 3-4 | 1,128 | 24 | 34 | 0.021 |
| add_word | strict | middle | 5-6 | 1,163 | 11 | 37 | 0.009 |
| add_word | strict | middle | 7-10 | 892 | 3 | 36 | 0.003 |
| add_word | strict | waqf | 11+ | 11 | 0 | 0 | 0.000 |
| add_word | strict | waqf | 3-4 | 21 | 0 | 0 | 0.000 |
| add_word | strict | waqf | 5-6 | 54 | 0 | 0 | 0.000 |
| add_word | strict | waqf | 7-10 | 74 | 0 | 0 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | first | 11+ | 28 | 0 | 0 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | first | 3-4 | 46 | 0 | 0 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | first | 5-6 | 196 | 0 | 2 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | first | 7-10 | 178 | 0 | 4 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | last | 11+ | 46 | 0 | 1 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | last | 3-4 | 26 | 0 | 7 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | last | 5-6 | 114 | 0 | 11 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | last | 7-10 | 560 | 0 | 41 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | middle | 1-2 | 21 | 0 | 0 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | middle | 11+ | 106 | 0 | 0 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | middle | 3-4 | 331 | 0 | 4 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | middle | 5-6 | 1,108 | 19 | 4 | 0.017 |
| consonant_pair:ت↔ط | balanced (shipped default) | middle | 7-10 | 1,080 | 13 | 4 | 0.012 |
| consonant_pair:ت↔ط | balanced (shipped default) | waqf | 11+ | 18 | 0 | 0 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | waqf | 3-4 | 6 | 0 | 0 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | waqf | 5-6 | 38 | 0 | 0 | 0.000 |
| consonant_pair:ت↔ط | balanced (shipped default) | waqf | 7-10 | 125 | 0 | 0 | 0.000 |
| consonant_pair:ت↔ط | strict | first | 11+ | 27 | 17 | 1 | 0.630 |
| consonant_pair:ت↔ط | strict | first | 3-4 | 46 | 45 | 0 | 0.978 |
| consonant_pair:ت↔ط | strict | first | 5-6 | 196 | 183 | 2 | 0.934 |
| consonant_pair:ت↔ط | strict | first | 7-10 | 178 | 149 | 4 | 0.837 |
| consonant_pair:ت↔ط | strict | last | 11+ | 46 | 43 | 1 | 0.935 |
| consonant_pair:ت↔ط | strict | last | 3-4 | 24 | 12 | 9 | 0.500 |
| consonant_pair:ت↔ط | strict | last | 5-6 | 112 | 68 | 13 | 0.607 |
| consonant_pair:ت↔ط | strict | last | 7-10 | 558 | 477 | 43 | 0.855 |
| consonant_pair:ت↔ط | strict | middle | 1-2 | 21 | 18 | 0 | 0.857 |
| consonant_pair:ت↔ط | strict | middle | 11+ | 104 | 58 | 2 | 0.558 |
| consonant_pair:ت↔ط | strict | middle | 3-4 | 318 | 187 | 17 | 0.588 |
| consonant_pair:ت↔ط | strict | middle | 5-6 | 1,073 | 660 | 39 | 0.615 |
| consonant_pair:ت↔ط | strict | middle | 7-10 | 1,069 | 624 | 15 | 0.584 |
| consonant_pair:ت↔ط | strict | waqf | 11+ | 18 | 11 | 0 | 0.611 |
| consonant_pair:ت↔ط | strict | waqf | 3-4 | 6 | 3 | 0 | 0.500 |
| consonant_pair:ت↔ط | strict | waqf | 5-6 | 38 | 22 | 0 | 0.579 |
| consonant_pair:ت↔ط | strict | waqf | 7-10 | 125 | 55 | 0 | 0.440 |
| consonant_pair:ح↔ه | balanced (shipped default) | first | 1-2 | 11 | 0 | 0 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | first | 11+ | 16 | 0 | 0 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | first | 3-4 | 125 | 0 | 0 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | first | 5-6 | 138 | 1 | 3 | 0.007 |
| consonant_pair:ح↔ه | balanced (shipped default) | first | 7-10 | 197 | 0 | 5 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | last | 11+ | 9 | 0 | 0 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | last | 3-4 | 69 | 0 | 4 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | last | 5-6 | 163 | 0 | 24 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | last | 7-10 | 268 | 0 | 9 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | middle | 1-2 | 209 | 1 | 7 | 0.005 |
| consonant_pair:ح↔ه | balanced (shipped default) | middle | 11+ | 83 | 0 | 0 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | middle | 3-4 | 683 | 0 | 7 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | middle | 5-6 | 1,324 | 62 | 8 | 0.047 |
| consonant_pair:ح↔ه | balanced (shipped default) | middle | 7-10 | 863 | 18 | 4 | 0.021 |
| consonant_pair:ح↔ه | balanced (shipped default) | waqf | 1-2 | 6 | 0 | 0 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | waqf | 11+ | 11 | 0 | 0 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | waqf | 3-4 | 33 | 0 | 0 | 0.000 |
| consonant_pair:ح↔ه | balanced (shipped default) | waqf | 5-6 | 94 | 3 | 1 | 0.032 |
| consonant_pair:ح↔ه | balanced (shipped default) | waqf | 7-10 | 114 | 4 | 0 | 0.035 |
| consonant_pair:ح↔ه | strict | first | 1-2 | 11 | 10 | 0 | 0.909 |
| consonant_pair:ح↔ه | strict | first | 11+ | 16 | 12 | 0 | 0.750 |
| consonant_pair:ح↔ه | strict | first | 3-4 | 125 | 124 | 0 | 0.992 |
| consonant_pair:ح↔ه | strict | first | 5-6 | 138 | 131 | 3 | 0.949 |
| consonant_pair:ح↔ه | strict | first | 7-10 | 196 | 162 | 6 | 0.827 |
| consonant_pair:ح↔ه | strict | last | 11+ | 9 | 2 | 0 | 0.222 |
| consonant_pair:ح↔ه | strict | last | 3-4 | 68 | 58 | 5 | 0.853 |
| consonant_pair:ح↔ه | strict | last | 5-6 | 153 | 100 | 34 | 0.654 |
| consonant_pair:ح↔ه | strict | last | 7-10 | 267 | 130 | 10 | 0.487 |
| consonant_pair:ح↔ه | strict | middle | 1-2 | 193 | 175 | 23 | 0.907 |
| consonant_pair:ح↔ه | strict | middle | 11+ | 83 | 39 | 0 | 0.470 |
| consonant_pair:ح↔ه | strict | middle | 3-4 | 659 | 476 | 31 | 0.722 |
| consonant_pair:ح↔ه | strict | middle | 5-6 | 1,292 | 705 | 40 | 0.546 |
| consonant_pair:ح↔ه | strict | middle | 7-10 | 850 | 499 | 17 | 0.587 |
| consonant_pair:ح↔ه | strict | waqf | 1-2 | 6 | 6 | 0 | 1.000 |
| consonant_pair:ح↔ه | strict | waqf | 11+ | 11 | 3 | 0 | 0.273 |
| consonant_pair:ح↔ه | strict | waqf | 3-4 | 33 | 14 | 0 | 0.424 |
| consonant_pair:ح↔ه | strict | waqf | 5-6 | 94 | 46 | 1 | 0.489 |
| consonant_pair:ح↔ه | strict | waqf | 7-10 | 114 | 69 | 0 | 0.605 |
| consonant_pair:ذ↔ز | balanced (shipped default) | first | 1-2 | 45 | 0 | 1 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | first | 11+ | 11 | 0 | 0 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | first | 3-4 | 102 | 0 | 1 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | first | 5-6 | 206 | 0 | 2 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | first | 7-10 | 196 | 0 | 1 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | last | 11+ | 8 | 0 | 0 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | last | 3-4 | 8 | 0 | 3 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | last | 5-6 | 71 | 0 | 1 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | last | 7-10 | 156 | 0 | 36 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | middle | 1-2 | 62 | 7 | 3 | 0.113 |
| consonant_pair:ذ↔ز | balanced (shipped default) | middle | 11+ | 21 | 0 | 0 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | middle | 3-4 | 279 | 2 | 2 | 0.007 |
| consonant_pair:ذ↔ز | balanced (shipped default) | middle | 5-6 | 1,224 | 0 | 6 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | middle | 7-10 | 476 | 0 | 1 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | waqf | 3-4 | 5 | 0 | 0 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | waqf | 5-6 | 29 | 0 | 0 | 0.000 |
| consonant_pair:ذ↔ز | balanced (shipped default) | waqf | 7-10 | 39 | 0 | 0 | 0.000 |
| consonant_pair:ذ↔ز | strict | first | 1-2 | 45 | 45 | 1 | 1.000 |
| consonant_pair:ذ↔ز | strict | first | 11+ | 11 | 11 | 0 | 1.000 |
| consonant_pair:ذ↔ز | strict | first | 3-4 | 102 | 100 | 1 | 0.980 |
| consonant_pair:ذ↔ز | strict | first | 5-6 | 206 | 200 | 2 | 0.971 |
| consonant_pair:ذ↔ز | strict | first | 7-10 | 196 | 188 | 1 | 0.959 |
| consonant_pair:ذ↔ز | strict | last | 11+ | 8 | 2 | 0 | 0.250 |
| consonant_pair:ذ↔ز | strict | last | 3-4 | 7 | 6 | 4 | 0.857 |
| consonant_pair:ذ↔ز | strict | last | 5-6 | 70 | 60 | 2 | 0.857 |
| consonant_pair:ذ↔ز | strict | last | 7-10 | 155 | 116 | 37 | 0.748 |
| consonant_pair:ذ↔ز | strict | middle | 1-2 | 62 | 23 | 3 | 0.371 |
| consonant_pair:ذ↔ز | strict | middle | 11+ | 21 | 19 | 0 | 0.905 |
| consonant_pair:ذ↔ز | strict | middle | 3-4 | 267 | 213 | 14 | 0.798 |
| consonant_pair:ذ↔ز | strict | middle | 5-6 | 1,206 | 1,016 | 24 | 0.842 |
| consonant_pair:ذ↔ز | strict | middle | 7-10 | 474 | 415 | 3 | 0.876 |
| consonant_pair:ذ↔ز | strict | waqf | 3-4 | 5 | 2 | 0 | 0.400 |
| consonant_pair:ذ↔ز | strict | waqf | 5-6 | 29 | 25 | 0 | 0.862 |
| consonant_pair:ذ↔ز | strict | waqf | 7-10 | 39 | 38 | 0 | 0.974 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | first | 1-2 | 44 | 44 | 1 | 1.000 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | first | 11+ | 4 | 4 | 0 | 1.000 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | first | 3-4 | 118 | 117 | 1 | 0.992 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | first | 5-6 | 226 | 220 | 2 | 0.973 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | first | 7-10 | 203 | 192 | 1 | 0.946 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | last | 11+ | 6 | 2 | 0 | 0.333 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | last | 3-4 | 7 | 4 | 0 | 0.571 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | last | 5-6 | 76 | 54 | 8 | 0.711 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | last | 7-10 | 137 | 100 | 33 | 0.730 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | middle | 1-2 | 59 | 25 | 4 | 0.424 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | middle | 11+ | 21 | 19 | 0 | 0.905 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | middle | 3-4 | 253 | 190 | 11 | 0.751 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | middle | 5-6 | 1,049 | 819 | 21 | 0.781 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | middle | 7-10 | 391 | 316 | 4 | 0.808 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | waqf | 11+ | 1 | 1 | 0 | 1.000 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | waqf | 3-4 | 11 | 2 | 0 | 0.182 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | waqf | 5-6 | 28 | 27 | 0 | 0.964 |
| consonant_pair:ذ↔ظ | balanced (shipped default) | waqf | 7-10 | 27 | 25 | 0 | 0.926 |
| consonant_pair:ذ↔ظ | strict | first | 1-2 | 44 | 44 | 1 | 1.000 |
| consonant_pair:ذ↔ظ | strict | first | 11+ | 4 | 4 | 0 | 1.000 |
| consonant_pair:ذ↔ظ | strict | first | 3-4 | 118 | 117 | 1 | 0.992 |
| consonant_pair:ذ↔ظ | strict | first | 5-6 | 226 | 220 | 2 | 0.973 |
| consonant_pair:ذ↔ظ | strict | first | 7-10 | 203 | 194 | 1 | 0.956 |
| consonant_pair:ذ↔ظ | strict | last | 11+ | 6 | 2 | 0 | 0.333 |
| consonant_pair:ذ↔ظ | strict | last | 3-4 | 6 | 5 | 1 | 0.833 |
| consonant_pair:ذ↔ظ | strict | last | 5-6 | 75 | 62 | 9 | 0.827 |
| consonant_pair:ذ↔ظ | strict | last | 7-10 | 137 | 103 | 33 | 0.752 |
| consonant_pair:ذ↔ظ | strict | middle | 1-2 | 59 | 25 | 4 | 0.424 |
| consonant_pair:ذ↔ظ | strict | middle | 11+ | 21 | 20 | 0 | 0.952 |
| consonant_pair:ذ↔ظ | strict | middle | 3-4 | 253 | 192 | 11 | 0.759 |
| consonant_pair:ذ↔ظ | strict | middle | 5-6 | 1,049 | 907 | 21 | 0.865 |
| consonant_pair:ذ↔ظ | strict | middle | 7-10 | 391 | 325 | 4 | 0.831 |
| consonant_pair:ذ↔ظ | strict | waqf | 11+ | 1 | 1 | 0 | 1.000 |
| consonant_pair:ذ↔ظ | strict | waqf | 3-4 | 11 | 2 | 0 | 0.182 |
| consonant_pair:ذ↔ظ | strict | waqf | 5-6 | 28 | 27 | 0 | 0.964 |
| consonant_pair:ذ↔ظ | strict | waqf | 7-10 | 27 | 26 | 0 | 0.963 |
| consonant_pair:س↔ش | balanced (shipped default) | first | 11+ | 9 | 8 | 0 | 0.889 |
| consonant_pair:س↔ش | balanced (shipped default) | first | 3-4 | 32 | 20 | 3 | 0.625 |
| consonant_pair:س↔ش | balanced (shipped default) | first | 5-6 | 76 | 57 | 1 | 0.750 |
| consonant_pair:س↔ش | balanced (shipped default) | first | 7-10 | 88 | 71 | 3 | 0.807 |
| consonant_pair:س↔ش | balanced (shipped default) | last | 11+ | 46 | 37 | 0 | 0.804 |
| consonant_pair:س↔ش | balanced (shipped default) | last | 3-4 | 30 | 30 | 4 | 1.000 |
| consonant_pair:س↔ش | balanced (shipped default) | last | 5-6 | 157 | 127 | 24 | 0.809 |
| consonant_pair:س↔ش | balanced (shipped default) | last | 7-10 | 610 | 446 | 21 | 0.731 |
| consonant_pair:س↔ش | balanced (shipped default) | middle | 1-2 | 16 | 16 | 2 | 1.000 |
| consonant_pair:س↔ش | balanced (shipped default) | middle | 11+ | 68 | 56 | 1 | 0.824 |
| consonant_pair:س↔ش | balanced (shipped default) | middle | 3-4 | 397 | 322 | 22 | 0.811 |
| consonant_pair:س↔ش | balanced (shipped default) | middle | 5-6 | 638 | 544 | 7 | 0.853 |
| consonant_pair:س↔ش | balanced (shipped default) | middle | 7-10 | 879 | 688 | 16 | 0.783 |
| consonant_pair:س↔ش | balanced (shipped default) | waqf | 11+ | 2 | 1 | 0 | 0.500 |
| consonant_pair:س↔ش | balanced (shipped default) | waqf | 3-4 | 15 | 12 | 0 | 0.800 |
| consonant_pair:س↔ش | balanced (shipped default) | waqf | 5-6 | 67 | 58 | 0 | 0.866 |
| consonant_pair:س↔ش | balanced (shipped default) | waqf | 7-10 | 129 | 106 | 3 | 0.822 |
| consonant_pair:س↔ش | strict | first | 11+ | 9 | 8 | 0 | 0.889 |
| consonant_pair:س↔ش | strict | first | 3-4 | 32 | 20 | 3 | 0.625 |
| consonant_pair:س↔ش | strict | first | 5-6 | 76 | 57 | 1 | 0.750 |
| consonant_pair:س↔ش | strict | first | 7-10 | 88 | 72 | 3 | 0.818 |
| consonant_pair:س↔ش | strict | last | 11+ | 46 | 37 | 0 | 0.804 |
| consonant_pair:س↔ش | strict | last | 3-4 | 30 | 30 | 4 | 1.000 |
| consonant_pair:س↔ش | strict | last | 5-6 | 156 | 140 | 25 | 0.897 |
| consonant_pair:س↔ش | strict | last | 7-10 | 610 | 454 | 21 | 0.744 |
| consonant_pair:س↔ش | strict | middle | 1-2 | 16 | 16 | 2 | 1.000 |
| consonant_pair:س↔ش | strict | middle | 11+ | 68 | 60 | 1 | 0.882 |
| consonant_pair:س↔ش | strict | middle | 3-4 | 397 | 343 | 22 | 0.864 |
| consonant_pair:س↔ش | strict | middle | 5-6 | 638 | 575 | 7 | 0.901 |
| consonant_pair:س↔ش | strict | middle | 7-10 | 879 | 705 | 16 | 0.802 |
| consonant_pair:س↔ش | strict | waqf | 11+ | 2 | 1 | 0 | 0.500 |
| consonant_pair:س↔ش | strict | waqf | 3-4 | 15 | 12 | 0 | 0.800 |
| consonant_pair:س↔ش | strict | waqf | 5-6 | 67 | 60 | 0 | 0.896 |
| consonant_pair:س↔ش | strict | waqf | 7-10 | 129 | 115 | 3 | 0.891 |
| consonant_pair:س↔ص | balanced (shipped default) | first | 11+ | 6 | 0 | 0 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | first | 3-4 | 45 | 0 | 2 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | first | 5-6 | 150 | 0 | 1 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | first | 7-10 | 102 | 0 | 6 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | last | 11+ | 49 | 0 | 0 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | last | 3-4 | 22 | 0 | 4 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | last | 5-6 | 173 | 1 | 15 | 0.006 |
| consonant_pair:س↔ص | balanced (shipped default) | last | 7-10 | 568 | 0 | 13 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | middle | 1-2 | 15 | 0 | 0 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | middle | 11+ | 57 | 0 | 0 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | middle | 3-4 | 349 | 0 | 7 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | middle | 5-6 | 664 | 0 | 3 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | middle | 7-10 | 1,000 | 0 | 2 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | waqf | 11+ | 5 | 0 | 0 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | waqf | 3-4 | 8 | 0 | 0 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | waqf | 5-6 | 50 | 0 | 0 | 0.000 |
| consonant_pair:س↔ص | balanced (shipped default) | waqf | 7-10 | 118 | 0 | 2 | 0.000 |
| consonant_pair:س↔ص | strict | first | 11+ | 6 | 6 | 0 | 1.000 |
| consonant_pair:س↔ص | strict | first | 3-4 | 45 | 41 | 2 | 0.911 |
| consonant_pair:س↔ص | strict | first | 5-6 | 150 | 144 | 1 | 0.960 |
| consonant_pair:س↔ص | strict | first | 7-10 | 102 | 101 | 6 | 0.990 |
| consonant_pair:س↔ص | strict | last | 11+ | 49 | 42 | 0 | 0.857 |
| consonant_pair:س↔ص | strict | last | 3-4 | 22 | 22 | 4 | 1.000 |
| consonant_pair:س↔ص | strict | last | 5-6 | 165 | 147 | 23 | 0.891 |
| consonant_pair:س↔ص | strict | last | 7-10 | 564 | 422 | 17 | 0.748 |
| consonant_pair:س↔ص | strict | middle | 1-2 | 13 | 13 | 2 | 1.000 |
| consonant_pair:س↔ص | strict | middle | 11+ | 57 | 51 | 0 | 0.895 |
| consonant_pair:س↔ص | strict | middle | 3-4 | 336 | 263 | 20 | 0.783 |
| consonant_pair:س↔ص | strict | middle | 5-6 | 650 | 529 | 17 | 0.814 |
| consonant_pair:س↔ص | strict | middle | 7-10 | 983 | 784 | 19 | 0.798 |
| consonant_pair:س↔ص | strict | waqf | 11+ | 5 | 3 | 0 | 0.600 |
| consonant_pair:س↔ص | strict | waqf | 3-4 | 8 | 7 | 0 | 0.875 |
| consonant_pair:س↔ص | strict | waqf | 5-6 | 50 | 45 | 0 | 0.900 |
| consonant_pair:س↔ص | strict | waqf | 7-10 | 117 | 105 | 3 | 0.897 |
| consonant_pair:ض↔ظ | balanced (shipped default) | first | 3-4 | 11 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | first | 5-6 | 15 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | first | 7-10 | 18 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | last | 11+ | 44 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | last | 3-4 | 1 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | last | 5-6 | 96 | 0 | 11 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | last | 7-10 | 133 | 0 | 8 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | middle | 1-2 | 3 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | middle | 11+ | 15 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | middle | 3-4 | 330 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | middle | 5-6 | 269 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | middle | 7-10 | 276 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | waqf | 3-4 | 28 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | waqf | 5-6 | 51 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | balanced (shipped default) | waqf | 7-10 | 20 | 0 | 0 | 0.000 |
| consonant_pair:ض↔ظ | strict | first | 3-4 | 11 | 11 | 0 | 1.000 |
| consonant_pair:ض↔ظ | strict | first | 5-6 | 15 | 15 | 0 | 1.000 |
| consonant_pair:ض↔ظ | strict | first | 7-10 | 18 | 17 | 0 | 0.944 |
| consonant_pair:ض↔ظ | strict | last | 11+ | 44 | 26 | 0 | 0.591 |
| consonant_pair:ض↔ظ | strict | last | 3-4 | 1 | 1 | 0 | 1.000 |
| consonant_pair:ض↔ظ | strict | last | 5-6 | 89 | 76 | 18 | 0.854 |
| consonant_pair:ض↔ظ | strict | last | 7-10 | 132 | 85 | 9 | 0.644 |
| consonant_pair:ض↔ظ | strict | middle | 1-2 | 3 | 2 | 0 | 0.667 |
| consonant_pair:ض↔ظ | strict | middle | 11+ | 15 | 11 | 0 | 0.733 |
| consonant_pair:ض↔ظ | strict | middle | 3-4 | 322 | 165 | 8 | 0.512 |
| consonant_pair:ض↔ظ | strict | middle | 5-6 | 265 | 184 | 4 | 0.694 |
| consonant_pair:ض↔ظ | strict | middle | 7-10 | 274 | 232 | 2 | 0.847 |
| consonant_pair:ض↔ظ | strict | waqf | 3-4 | 28 | 5 | 0 | 0.179 |
| consonant_pair:ض↔ظ | strict | waqf | 5-6 | 51 | 11 | 0 | 0.216 |
| consonant_pair:ض↔ظ | strict | waqf | 7-10 | 20 | 18 | 0 | 0.900 |
| consonant_pair:ط↔ق | balanced (shipped default) | first | 1-2 | 104 | 1 | 15 | 0.010 |
| consonant_pair:ط↔ق | balanced (shipped default) | first | 11+ | 6 | 6 | 0 | 1.000 |
| consonant_pair:ط↔ق | balanced (shipped default) | first | 3-4 | 150 | 42 | 0 | 0.280 |
| consonant_pair:ط↔ق | balanced (shipped default) | first | 5-6 | 230 | 200 | 1 | 0.870 |
| consonant_pair:ط↔ق | balanced (shipped default) | first | 7-10 | 102 | 87 | 5 | 0.853 |
| consonant_pair:ط↔ق | balanced (shipped default) | last | 11+ | 21 | 5 | 0 | 0.238 |
| consonant_pair:ط↔ق | balanced (shipped default) | last | 3-4 | 38 | 22 | 7 | 0.579 |
| consonant_pair:ط↔ق | balanced (shipped default) | last | 5-6 | 144 | 84 | 22 | 0.583 |
| consonant_pair:ط↔ق | balanced (shipped default) | last | 7-10 | 365 | 161 | 11 | 0.441 |
| consonant_pair:ط↔ق | balanced (shipped default) | middle | 1-2 | 36 | 30 | 0 | 0.833 |
| consonant_pair:ط↔ق | balanced (shipped default) | middle | 11+ | 34 | 26 | 0 | 0.765 |
| consonant_pair:ط↔ق | balanced (shipped default) | middle | 3-4 | 439 | 322 | 17 | 0.733 |
| consonant_pair:ط↔ق | balanced (shipped default) | middle | 5-6 | 690 | 563 | 27 | 0.816 |
| consonant_pair:ط↔ق | balanced (shipped default) | middle | 7-10 | 600 | 521 | 7 | 0.868 |
| consonant_pair:ط↔ق | balanced (shipped default) | waqf | 11+ | 11 | 1 | 0 | 0.091 |
| consonant_pair:ط↔ق | balanced (shipped default) | waqf | 3-4 | 28 | 18 | 0 | 0.643 |
| consonant_pair:ط↔ق | balanced (shipped default) | waqf | 5-6 | 46 | 29 | 0 | 0.630 |
| consonant_pair:ط↔ق | balanced (shipped default) | waqf | 7-10 | 89 | 70 | 0 | 0.787 |
| consonant_pair:ط↔ق | strict | first | 1-2 | 104 | 1 | 15 | 0.010 |
| consonant_pair:ط↔ق | strict | first | 11+ | 6 | 6 | 0 | 1.000 |
| consonant_pair:ط↔ق | strict | first | 3-4 | 150 | 47 | 0 | 0.313 |
| consonant_pair:ط↔ق | strict | first | 5-6 | 230 | 201 | 1 | 0.874 |
| consonant_pair:ط↔ق | strict | first | 7-10 | 102 | 87 | 5 | 0.853 |
| consonant_pair:ط↔ق | strict | last | 11+ | 21 | 11 | 0 | 0.524 |
| consonant_pair:ط↔ق | strict | last | 3-4 | 38 | 23 | 7 | 0.605 |
| consonant_pair:ط↔ق | strict | last | 5-6 | 142 | 105 | 24 | 0.739 |
| consonant_pair:ط↔ق | strict | last | 7-10 | 365 | 173 | 11 | 0.474 |
| consonant_pair:ط↔ق | strict | middle | 1-2 | 36 | 30 | 0 | 0.833 |
| consonant_pair:ط↔ق | strict | middle | 11+ | 34 | 28 | 0 | 0.824 |
| consonant_pair:ط↔ق | strict | middle | 3-4 | 439 | 329 | 17 | 0.749 |
| consonant_pair:ط↔ق | strict | middle | 5-6 | 690 | 589 | 27 | 0.854 |
| consonant_pair:ط↔ق | strict | middle | 7-10 | 600 | 538 | 7 | 0.897 |
| consonant_pair:ط↔ق | strict | waqf | 11+ | 11 | 2 | 0 | 0.182 |
| consonant_pair:ط↔ق | strict | waqf | 3-4 | 28 | 18 | 0 | 0.643 |
| consonant_pair:ط↔ق | strict | waqf | 5-6 | 46 | 29 | 0 | 0.630 |
| consonant_pair:ط↔ق | strict | waqf | 7-10 | 89 | 73 | 0 | 0.820 |
| consonant_pair:ق↔ك | balanced (shipped default) | first | 1-2 | 82 | 2 | 4 | 0.024 |
| consonant_pair:ق↔ك | balanced (shipped default) | first | 11+ | 10 | 0 | 0 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | first | 3-4 | 117 | 1 | 0 | 0.009 |
| consonant_pair:ق↔ك | balanced (shipped default) | first | 5-6 | 285 | 5 | 4 | 0.018 |
| consonant_pair:ق↔ك | balanced (shipped default) | first | 7-10 | 186 | 1 | 5 | 0.005 |
| consonant_pair:ق↔ك | balanced (shipped default) | last | 11+ | 15 | 0 | 0 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | last | 3-4 | 27 | 0 | 8 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | last | 5-6 | 167 | 0 | 15 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | last | 7-10 | 415 | 0 | 29 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | middle | 1-2 | 99 | 1 | 1 | 0.010 |
| consonant_pair:ق↔ك | balanced (shipped default) | middle | 11+ | 48 | 0 | 1 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | middle | 3-4 | 785 | 0 | 9 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | middle | 5-6 | 1,109 | 37 | 6 | 0.033 |
| consonant_pair:ق↔ك | balanced (shipped default) | middle | 7-10 | 840 | 11 | 8 | 0.013 |
| consonant_pair:ق↔ك | balanced (shipped default) | waqf | 11+ | 3 | 0 | 0 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | waqf | 3-4 | 28 | 0 | 0 | 0.000 |
| consonant_pair:ق↔ك | balanced (shipped default) | waqf | 5-6 | 66 | 1 | 0 | 0.015 |
| consonant_pair:ق↔ك | balanced (shipped default) | waqf | 7-10 | 79 | 2 | 0 | 0.025 |
| consonant_pair:ق↔ك | strict | first | 1-2 | 82 | 81 | 4 | 0.988 |
| consonant_pair:ق↔ك | strict | first | 11+ | 10 | 7 | 0 | 0.700 |
| consonant_pair:ق↔ك | strict | first | 3-4 | 117 | 114 | 0 | 0.974 |
| consonant_pair:ق↔ك | strict | first | 5-6 | 285 | 266 | 4 | 0.933 |
| consonant_pair:ق↔ك | strict | first | 7-10 | 186 | 178 | 5 | 0.957 |
| consonant_pair:ق↔ك | strict | last | 11+ | 15 | 6 | 0 | 0.400 |
| consonant_pair:ق↔ك | strict | last | 3-4 | 26 | 11 | 9 | 0.423 |
| consonant_pair:ق↔ك | strict | last | 5-6 | 164 | 122 | 18 | 0.744 |
| consonant_pair:ق↔ك | strict | last | 7-10 | 413 | 241 | 31 | 0.584 |
| consonant_pair:ق↔ك | strict | middle | 1-2 | 95 | 55 | 5 | 0.579 |
| consonant_pair:ق↔ك | strict | middle | 11+ | 48 | 33 | 1 | 0.688 |
| consonant_pair:ق↔ك | strict | middle | 3-4 | 757 | 555 | 37 | 0.733 |
| consonant_pair:ق↔ك | strict | middle | 5-6 | 1,080 | 896 | 35 | 0.830 |
| consonant_pair:ق↔ك | strict | middle | 7-10 | 808 | 586 | 40 | 0.725 |
| consonant_pair:ق↔ك | strict | waqf | 11+ | 3 | 1 | 0 | 0.333 |
| consonant_pair:ق↔ك | strict | waqf | 3-4 | 28 | 16 | 0 | 0.571 |
| consonant_pair:ق↔ك | strict | waqf | 5-6 | 66 | 41 | 0 | 0.621 |
| consonant_pair:ق↔ك | strict | waqf | 7-10 | 78 | 64 | 1 | 0.821 |
| consonant_random | balanced (shipped default) | first | 1-2 | 58 | 37 | 4 | 0.638 |
| consonant_random | balanced (shipped default) | first | 11+ | 19 | 14 | 0 | 0.737 |
| consonant_random | balanced (shipped default) | first | 3-4 | 211 | 142 | 8 | 0.673 |
| consonant_random | balanced (shipped default) | first | 5-6 | 371 | 250 | 12 | 0.674 |
| consonant_random | balanced (shipped default) | first | 7-10 | 271 | 187 | 12 | 0.690 |
| consonant_random | balanced (shipped default) | last | 11+ | 38 | 15 | 1 | 0.395 |
| consonant_random | balanced (shipped default) | last | 3-4 | 61 | 37 | 7 | 0.607 |
| consonant_random | balanced (shipped default) | last | 5-6 | 299 | 157 | 46 | 0.525 |
| consonant_random | balanced (shipped default) | last | 7-10 | 507 | 273 | 34 | 0.538 |
| consonant_random | balanced (shipped default) | middle | 1-2 | 229 | 148 | 25 | 0.646 |
| consonant_random | balanced (shipped default) | middle | 11+ | 56 | 29 | 1 | 0.518 |
| consonant_random | balanced (shipped default) | middle | 3-4 | 1,037 | 755 | 91 | 0.728 |
| consonant_random | balanced (shipped default) | middle | 5-6 | 1,229 | 822 | 53 | 0.669 |
| consonant_random | balanced (shipped default) | middle | 7-10 | 849 | 534 | 31 | 0.629 |
| consonant_random | balanced (shipped default) | waqf | 1-2 | 1 | 0 | 0 | 0.000 |
| consonant_random | balanced (shipped default) | waqf | 11+ | 11 | 7 | 0 | 0.636 |
| consonant_random | balanced (shipped default) | waqf | 3-4 | 30 | 21 | 0 | 0.700 |
| consonant_random | balanced (shipped default) | waqf | 5-6 | 62 | 43 | 0 | 0.694 |
| consonant_random | balanced (shipped default) | waqf | 7-10 | 76 | 47 | 2 | 0.618 |
| consonant_random | strict | first | 1-2 | 58 | 37 | 4 | 0.638 |
| consonant_random | strict | first | 11+ | 19 | 15 | 0 | 0.789 |
| consonant_random | strict | first | 3-4 | 211 | 170 | 8 | 0.806 |
| consonant_random | strict | first | 5-6 | 371 | 254 | 12 | 0.685 |
| consonant_random | strict | first | 7-10 | 271 | 191 | 12 | 0.705 |
| consonant_random | strict | last | 11+ | 38 | 15 | 1 | 0.395 |
| consonant_random | strict | last | 3-4 | 61 | 41 | 7 | 0.672 |
| consonant_random | strict | last | 5-6 | 298 | 183 | 47 | 0.614 |
| consonant_random | strict | last | 7-10 | 505 | 291 | 36 | 0.576 |
| consonant_random | strict | middle | 1-2 | 229 | 148 | 25 | 0.646 |
| consonant_random | strict | middle | 11+ | 56 | 33 | 1 | 0.589 |
| consonant_random | strict | middle | 3-4 | 1,036 | 784 | 92 | 0.757 |
| consonant_random | strict | middle | 5-6 | 1,228 | 854 | 54 | 0.695 |
| consonant_random | strict | middle | 7-10 | 849 | 580 | 31 | 0.683 |
| consonant_random | strict | waqf | 1-2 | 1 | 0 | 0 | 0.000 |
| consonant_random | strict | waqf | 11+ | 11 | 7 | 0 | 0.636 |
| consonant_random | strict | waqf | 3-4 | 30 | 22 | 0 | 0.733 |
| consonant_random | strict | waqf | 5-6 | 62 | 44 | 0 | 0.710 |
| consonant_random | strict | waqf | 7-10 | 76 | 51 | 2 | 0.671 |
| haraka_drop | balanced (shipped default) | first | 1-2 | 61 | 40 | 0 | 0.656 |
| haraka_drop | balanced (shipped default) | first | 11+ | 16 | 10 | 0 | 0.625 |
| haraka_drop | balanced (shipped default) | first | 3-4 | 209 | 137 | 3 | 0.656 |
| haraka_drop | balanced (shipped default) | first | 5-6 | 378 | 201 | 7 | 0.532 |
| haraka_drop | balanced (shipped default) | first | 7-10 | 290 | 173 | 5 | 0.597 |
| haraka_drop | balanced (shipped default) | last | 11+ | 68 | 29 | 0 | 0.426 |
| haraka_drop | balanced (shipped default) | last | 3-4 | 61 | 42 | 1 | 0.689 |
| haraka_drop | balanced (shipped default) | last | 5-6 | 312 | 175 | 17 | 0.561 |
| haraka_drop | balanced (shipped default) | last | 7-10 | 541 | 251 | 15 | 0.464 |
| haraka_drop | balanced (shipped default) | middle | 1-2 | 268 | 109 | 11 | 0.407 |
| haraka_drop | balanced (shipped default) | middle | 11+ | 55 | 28 | 0 | 0.509 |
| haraka_drop | balanced (shipped default) | middle | 3-4 | 1,102 | 611 | 59 | 0.554 |
| haraka_drop | balanced (shipped default) | middle | 5-6 | 1,189 | 649 | 47 | 0.546 |
| haraka_drop | balanced (shipped default) | middle | 7-10 | 829 | 411 | 28 | 0.496 |
| haraka_drop | balanced (shipped default) | waqf | 1-2 | 2 | 2 | 0 | 1.000 |
| haraka_drop | balanced (shipped default) | waqf | 11+ | 6 | 1 | 0 | 0.167 |
| haraka_drop | balanced (shipped default) | waqf | 3-4 | 36 | 15 | 1 | 0.417 |
| haraka_drop | balanced (shipped default) | waqf | 5-6 | 60 | 33 | 0 | 0.550 |
| haraka_drop | balanced (shipped default) | waqf | 7-10 | 65 | 32 | 0 | 0.492 |
| haraka_drop | strict | first | 1-2 | 61 | 40 | 0 | 0.656 |
| haraka_drop | strict | first | 11+ | 16 | 11 | 0 | 0.688 |
| haraka_drop | strict | first | 3-4 | 209 | 140 | 3 | 0.670 |
| haraka_drop | strict | first | 5-6 | 378 | 251 | 7 | 0.664 |
| haraka_drop | strict | first | 7-10 | 290 | 196 | 5 | 0.676 |
| haraka_drop | strict | last | 11+ | 68 | 35 | 0 | 0.515 |
| haraka_drop | strict | last | 3-4 | 59 | 43 | 3 | 0.729 |
| haraka_drop | strict | last | 5-6 | 311 | 195 | 18 | 0.627 |
| haraka_drop | strict | last | 7-10 | 539 | 290 | 17 | 0.538 |
| haraka_drop | strict | middle | 1-2 | 268 | 109 | 11 | 0.407 |
| haraka_drop | strict | middle | 11+ | 55 | 31 | 0 | 0.564 |
| haraka_drop | strict | middle | 3-4 | 1,096 | 646 | 65 | 0.589 |
| haraka_drop | strict | middle | 5-6 | 1,186 | 717 | 50 | 0.605 |
| haraka_drop | strict | middle | 7-10 | 828 | 481 | 29 | 0.581 |
| haraka_drop | strict | waqf | 1-2 | 2 | 2 | 0 | 1.000 |
| haraka_drop | strict | waqf | 11+ | 6 | 1 | 0 | 0.167 |
| haraka_drop | strict | waqf | 3-4 | 36 | 15 | 1 | 0.417 |
| haraka_drop | strict | waqf | 5-6 | 60 | 36 | 0 | 0.600 |
| haraka_drop | strict | waqf | 7-10 | 65 | 35 | 0 | 0.538 |
| haraka_swap | balanced (shipped default) | first | 1-2 | 67 | 67 | 4 | 1.000 |
| haraka_swap | balanced (shipped default) | first | 11+ | 17 | 16 | 1 | 0.941 |
| haraka_swap | balanced (shipped default) | first | 3-4 | 219 | 207 | 1 | 0.945 |
| haraka_swap | balanced (shipped default) | first | 5-6 | 378 | 351 | 10 | 0.929 |
| haraka_swap | balanced (shipped default) | first | 7-10 | 272 | 247 | 10 | 0.908 |
| haraka_swap | balanced (shipped default) | last | 11+ | 43 | 26 | 0 | 0.605 |
| haraka_swap | balanced (shipped default) | last | 3-4 | 60 | 52 | 2 | 0.867 |
| haraka_swap | balanced (shipped default) | last | 5-6 | 307 | 201 | 17 | 0.655 |
| haraka_swap | balanced (shipped default) | last | 7-10 | 527 | 321 | 14 | 0.609 |
| haraka_swap | balanced (shipped default) | middle | 1-2 | 255 | 159 | 21 | 0.624 |
| haraka_swap | balanced (shipped default) | middle | 11+ | 49 | 30 | 1 | 0.612 |
| haraka_swap | balanced (shipped default) | middle | 3-4 | 1,074 | 751 | 73 | 0.699 |
| haraka_swap | balanced (shipped default) | middle | 5-6 | 1,219 | 893 | 42 | 0.733 |
| haraka_swap | balanced (shipped default) | middle | 7-10 | 855 | 600 | 28 | 0.702 |
| haraka_swap | balanced (shipped default) | waqf | 11+ | 7 | 4 | 0 | 0.571 |
| haraka_swap | balanced (shipped default) | waqf | 3-4 | 29 | 13 | 0 | 0.448 |
| haraka_swap | balanced (shipped default) | waqf | 5-6 | 58 | 41 | 0 | 0.707 |
| haraka_swap | balanced (shipped default) | waqf | 7-10 | 82 | 53 | 0 | 0.646 |
| haraka_swap | strict | first | 1-2 | 67 | 67 | 4 | 1.000 |
| haraka_swap | strict | first | 11+ | 17 | 16 | 1 | 0.941 |
| haraka_swap | strict | first | 3-4 | 219 | 209 | 1 | 0.954 |
| haraka_swap | strict | first | 5-6 | 378 | 351 | 10 | 0.929 |
| haraka_swap | strict | first | 7-10 | 272 | 251 | 10 | 0.923 |
| haraka_swap | strict | last | 11+ | 43 | 26 | 0 | 0.605 |
| haraka_swap | strict | last | 3-4 | 59 | 54 | 3 | 0.915 |
| haraka_swap | strict | last | 5-6 | 306 | 235 | 18 | 0.768 |
| haraka_swap | strict | last | 7-10 | 527 | 344 | 14 | 0.653 |
| haraka_swap | strict | middle | 1-2 | 255 | 159 | 21 | 0.624 |
| haraka_swap | strict | middle | 11+ | 49 | 33 | 1 | 0.673 |
| haraka_swap | strict | middle | 3-4 | 1,074 | 763 | 73 | 0.710 |
| haraka_swap | strict | middle | 5-6 | 1,219 | 930 | 42 | 0.763 |
| haraka_swap | strict | middle | 7-10 | 855 | 646 | 28 | 0.756 |
| haraka_swap | strict | waqf | 11+ | 7 | 4 | 0 | 0.571 |
| haraka_swap | strict | waqf | 3-4 | 29 | 14 | 0 | 0.483 |
| haraka_swap | strict | waqf | 5-6 | 58 | 42 | 0 | 0.724 |
| haraka_swap | strict | waqf | 7-10 | 82 | 57 | 0 | 0.695 |
| shaddah_drop | balanced (shipped default) | first | 11+ | 28 | 21 | 0 | 0.750 |
| shaddah_drop | balanced (shipped default) | first | 3-4 | 56 | 4 | 0 | 0.071 |
| shaddah_drop | balanced (shipped default) | first | 5-6 | 391 | 13 | 9 | 0.033 |
| shaddah_drop | balanced (shipped default) | first | 7-10 | 380 | 155 | 8 | 0.408 |
| shaddah_drop | balanced (shipped default) | last | 11+ | 79 | 2 | 0 | 0.025 |
| shaddah_drop | balanced (shipped default) | last | 3-4 | 2 | 0 | 0 | 0.000 |
| shaddah_drop | balanced (shipped default) | last | 5-6 | 177 | 4 | 27 | 0.023 |
| shaddah_drop | balanced (shipped default) | last | 7-10 | 315 | 15 | 20 | 0.048 |
| shaddah_drop | balanced (shipped default) | middle | 11+ | 79 | 13 | 0 | 0.165 |
| shaddah_drop | balanced (shipped default) | middle | 3-4 | 647 | 4 | 4 | 0.006 |
| shaddah_drop | balanced (shipped default) | middle | 5-6 | 1,682 | 20 | 8 | 0.012 |
| shaddah_drop | balanced (shipped default) | middle | 7-10 | 1,013 | 228 | 15 | 0.225 |
| shaddah_drop | balanced (shipped default) | waqf | 11+ | 9 | 1 | 0 | 0.111 |
| shaddah_drop | balanced (shipped default) | waqf | 3-4 | 12 | 0 | 0 | 0.000 |
| shaddah_drop | balanced (shipped default) | waqf | 5-6 | 92 | 0 | 0 | 0.000 |
| shaddah_drop | balanced (shipped default) | waqf | 7-10 | 92 | 14 | 0 | 0.152 |
| shaddah_drop | strict | first | 11+ | 28 | 28 | 0 | 1.000 |
| shaddah_drop | strict | first | 3-4 | 56 | 47 | 0 | 0.839 |
| shaddah_drop | strict | first | 5-6 | 391 | 382 | 9 | 0.977 |
| shaddah_drop | strict | first | 7-10 | 378 | 358 | 10 | 0.947 |
| shaddah_drop | strict | last | 11+ | 79 | 4 | 0 | 0.051 |
| shaddah_drop | strict | last | 3-4 | 2 | 0 | 0 | 0.000 |
| shaddah_drop | strict | last | 5-6 | 177 | 59 | 27 | 0.333 |
| shaddah_drop | strict | last | 7-10 | 313 | 141 | 22 | 0.450 |
| shaddah_drop | strict | middle | 11+ | 79 | 51 | 0 | 0.646 |
| shaddah_drop | strict | middle | 3-4 | 619 | 340 | 32 | 0.549 |
| shaddah_drop | strict | middle | 5-6 | 1,656 | 668 | 34 | 0.403 |
| shaddah_drop | strict | middle | 7-10 | 999 | 546 | 29 | 0.547 |
| shaddah_drop | strict | waqf | 11+ | 9 | 4 | 0 | 0.444 |
| shaddah_drop | strict | waqf | 3-4 | 12 | 5 | 0 | 0.417 |
| shaddah_drop | strict | waqf | 5-6 | 92 | 29 | 0 | 0.315 |
| shaddah_drop | strict | waqf | 7-10 | 92 | 40 | 0 | 0.435 |
| skip_word | balanced (shipped default) | first | 1-2 | 0 | 0 | 70 | — |
| skip_word | balanced (shipped default) | first | 11+ | 0 | 0 | 14 | — |
| skip_word | balanced (shipped default) | first | 3-4 | 0 | 0 | 240 | — |
| skip_word | balanced (shipped default) | first | 5-6 | 0 | 0 | 373 | — |
| skip_word | balanced (shipped default) | first | 7-10 | 1 | 1 | 278 | 1.000 |
| skip_word | balanced (shipped default) | last | 11+ | 0 | 0 | 46 | — |
| skip_word | balanced (shipped default) | last | 3-4 | 0 | 0 | 52 | — |
| skip_word | balanced (shipped default) | last | 5-6 | 0 | 0 | 309 | — |
| skip_word | balanced (shipped default) | last | 7-10 | 0 | 0 | 535 | — |
| skip_word | balanced (shipped default) | middle | 1-2 | 223 | 218 | 65 | 0.978 |
| skip_word | balanced (shipped default) | middle | 11+ | 57 | 57 | 13 | 1.000 |
| skip_word | balanced (shipped default) | middle | 3-4 | 794 | 793 | 332 | 0.999 |
| skip_word | balanced (shipped default) | middle | 5-6 | 983 | 983 | 286 | 1.000 |
| skip_word | balanced (shipped default) | middle | 7-10 | 676 | 675 | 224 | 0.999 |
| skip_word | balanced (shipped default) | waqf | 1-2 | 2 | 2 | 0 | 1.000 |
| skip_word | balanced (shipped default) | waqf | 11+ | 4 | 4 | 1 | 1.000 |
| skip_word | balanced (shipped default) | waqf | 3-4 | 26 | 26 | 1 | 1.000 |
| skip_word | balanced (shipped default) | waqf | 5-6 | 59 | 58 | 5 | 0.983 |
| skip_word | balanced (shipped default) | waqf | 7-10 | 64 | 64 | 5 | 1.000 |
| skip_word | strict | first | 1-2 | 0 | 0 | 70 | — |
| skip_word | strict | first | 11+ | 0 | 0 | 14 | — |
| skip_word | strict | first | 3-4 | 0 | 0 | 240 | — |
| skip_word | strict | first | 5-6 | 0 | 0 | 373 | — |
| skip_word | strict | first | 7-10 | 1 | 1 | 278 | 1.000 |
| skip_word | strict | last | 11+ | 0 | 0 | 46 | — |
| skip_word | strict | last | 3-4 | 0 | 0 | 52 | — |
| skip_word | strict | last | 5-6 | 0 | 0 | 309 | — |
| skip_word | strict | last | 7-10 | 0 | 0 | 535 | — |
| skip_word | strict | middle | 1-2 | 223 | 218 | 65 | 0.978 |
| skip_word | strict | middle | 11+ | 57 | 57 | 13 | 1.000 |
| skip_word | strict | middle | 3-4 | 794 | 793 | 332 | 0.999 |
| skip_word | strict | middle | 5-6 | 983 | 983 | 286 | 1.000 |
| skip_word | strict | middle | 7-10 | 676 | 675 | 224 | 0.999 |
| skip_word | strict | waqf | 1-2 | 2 | 2 | 0 | 1.000 |
| skip_word | strict | waqf | 11+ | 4 | 4 | 1 | 1.000 |
| skip_word | strict | waqf | 3-4 | 26 | 26 | 1 | 1.000 |
| skip_word | strict | waqf | 5-6 | 59 | 58 | 5 | 0.983 |
| skip_word | strict | waqf | 7-10 | 64 | 64 | 5 | 1.000 |

Kinds no clip could carry, with the number of clips that could not: consonant_pair:ت↔ط (1,637), consonant_pair:ح↔ه (1,254), consonant_pair:ذ↔ز (2,747), consonant_pair:ذ↔ظ (2,995), consonant_pair:س↔ش (2,376), consonant_pair:س↔ص (2,306), consonant_pair:ض↔ظ (4,413), consonant_pair:ط↔ق (2,497), consonant_pair:ق↔ك (1,291), shaddah_drop (597), skip_word (4).

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

## Evidence

`tools/tadabur/followalong_audit/oracle_sweep/clean_findings.jsonl.gz` holds every false flag from both clean sweeps, and `tools/tadabur/followalong_audit/oracle_sweep/planted_rows.jsonl.gz` every planted row, each with its per-check statuses, its position and whether a window boundary splits it. Both are compressed, sorted-key JSONL, regenerated deterministically with this report.

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

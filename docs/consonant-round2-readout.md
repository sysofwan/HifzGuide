# The tlog consonant round's readout (#227)

clip-level split, same users possible on both sides

## In plain language

The round **fails** under the frozen decision rule, so **C95 stays**. The pre-claim gates all
passed. The candidate is clearly better than C95 at avoiding false flags overall and on
consonants, and agrees more with the teacher, but it fails one no-harm guard: **G4**, the
tashkeel-caused false flags. The candidate's point estimate is +0.19 per 1,000 words above C95's,
and the paired 95% upper bound is **+1.53 per 1,000**, above the frozen ceiling of **≤ +1.0** — over it by
**0.53 per 1,000**.
The rule is "any gate fails → fail"; no number here can be traded against another.

Headline bars on the pooled confirm.1+confirm.2 gating set (5,742 gating clips, 57,935 graded
words). The two frozen tranches hold 5,762 gating clips: **20 are ungradable** because the
phonetizer cannot build their ayah references, including **all 14 surah-106 rows**, so surah
106 drops out and the clustered bootstrap runs on **113 surah clusters**, not 114. The 20
clips are named with their ayat in the coverage section below.

- all-cause false flags, candidate − C95: **−54.5 per 1,000** (95% [−63.6, −47.4]) — pass;
- consonant-caused, candidate − C95: **−57.9 per 1,000** (95% [−65.0, −52.2]) — pass;
- bar 2's reduction: **54.5 per 1,000** against the required **≥ 4.7** — pass.

Candidate − control is **−43.2 per 1,000** (95% [−53.0, −35.7]). That is one training run per
arm (n = 1, no seed replicate): descriptive, **never a test**, and it never gates.

Shaddah is **reported only**, never gating (owner, 2026-10-28): shaddah-caused false flags,
candidate − C95, are **−0.05 per 1,000** (95% [−0.59, 0.42]); shaddah agreement with the
teacher is **+5.37 pt** (95% [4.52, 6.42]); the qdat_bench shaddah arm is in the pre-claim
section above.

Pre-claim verdict: pass

## The pre-claim gates

- bar4a.qdat: pass — word agreement with the teacher on qdat: paired 95% lower bound > 0 (point 5.905, bound 4.835, threshold 0.000)
- bar4b.qdat: pass — the same on the teacher-departure words of qdat: paired 95% lower bound ≥ −0.5 pt (point 6.952, bound 5.434, threshold -0.500)
- bar5.qdat: pass — qdat all-cause false flags per 1,000: candidate point estimate < C95's
- catch_loss: pass — the catch-loss audit: at most 20% of the decisive removed verdicts are real mistakes, and at least 80 decisive removed verdicts are heard
- g1.qud_other: pass — QUD other half, all-cause false flags per 1,000: candidate − C95 paired 95% upper bound ≤ +1.0 (point -0.197, bound 0.177, threshold 1.000)
- g2.tadabur_dev_unseen: pass — Tadabur dev, decode agreement with the teacher at stream_b0: paired 95% lower bound ≥ −1.0 pt (point -0.180, bound -0.282, threshold -1.000)
- g3.qdat_bench.consonant: pass — qdat_bench stream arm, consonant agreement: paired 95% lower bound ≥ −2.0 pt (point 3.029, bound 2.150, threshold -2.000)
- g3.qdat_bench.haraka: pass — qdat_bench stream arm, haraka agreement: paired 95% lower bound ≥ −2.0 pt (point 2.258, bound 1.295, threshold -2.000)
- g6.ship_a_verified: pass — Ship A held-out verified mistakes at ط↔ق and س↔ش: candidate flags ≥ C95 − 2 sites (paired count) (threshold -2.000, difference 1)

Reported beside the gates, never folded into them:

- g3.qdat_bench.shaddah: pass — qdat_bench stream arm, shaddah agreement: paired 95% lower bound ≥ −2.0 pt (95% [-0.265, 2.486], point 1.096, bound -0.265, threshold -2.000)

## The control's columns

{"qdat": {"all_cause_per_1000": 50.60975609756098}, "qdat_bench": {"consonant": 88.89228886168911, "haraka": 89.30421909696521, "shaddah": 92.32876712328768}, "qud_other": {"all_cause_per_1000": 3.2847194849559846}, "tadabur_dev_unseen": {"teacher_agreement_points": 94.87665741597286}}

## The pooled confirm verdict

Verdict: fail (pre-claim pass)

### Bars

- bar1.all_cause: pass — the pooled tlog confirm.1+2 gating set, all-cause false flags per 1,000 (shipped default): candidate − C95 paired 95% upper bound < 0 (point -54.467, bound -47.404, threshold 0.000)
- bar2.reduction: pass — the same difference as a point reduction of at least 4.7 per 1,000 (reduction 54.467, required 4.700)
- bar3.consonant_cause: pass — the pooled tlog confirm.1+2 gating set, consonant-caused false flags per 1,000: candidate − C95 paired 95% upper bound < 0 (point -57.880, bound -52.166, threshold 0.000)
- bar4a.tlog_confirm: pass — word agreement with the teacher on the pooled tlog confirm tranches: paired 95% lower bound > 0 (point 11.405, bound 10.398, threshold 0.000)
- bar4b.tlog_confirm: pass — the same on the teacher-departure words of the pooled tlog confirm tranches: paired 95% lower bound ≥ −0.5 pt (point 10.887, bound 9.815, threshold -0.500)
- g4.tashkeel_cause: fail — the pooled tlog confirm.1+2 gating set, tashkeel-caused false flags per 1,000: candidate − C95 paired 95% upper bound ≤ +1.0 (point 0.187, bound 1.527, threshold 1.000)
- g5.ء→∅: pass — no substitution's paired 99.5% lower bound > 0 (bound -5.537)
- g5.د→ت: pass — no substitution's paired 99.5% lower bound > 0 (bound -2.318)
- g5.ر→ل: pass — no substitution's paired 99.5% lower bound > 0 (bound -3.115)
- g5.ر→∅: pass — no substitution's paired 99.5% lower bound > 0 (bound -2.836)
- g5.ل→ن: pass — no substitution's paired 99.5% lower bound > 0 (bound -2.613)
- g5.ل→∅: pass — no substitution's paired 99.5% lower bound > 0 (bound -2.024)
- g5.م→∅: pass — no substitution's paired 99.5% lower bound > 0 (bound -2.211)
- g5.ن→∅: pass — no substitution's paired 99.5% lower bound > 0 (bound -1.642)
- g5.ه→∅: pass — no substitution's paired 99.5% lower bound > 0 (bound -1.773)
- g5.و→∅: pass — no substitution's paired 99.5% lower bound > 0 (bound -1.738)
- g5.ي→∅: pass — no substitution's paired 99.5% lower bound > 0 (bound -1.812)

### Reproduction check (not a bar)

- c178_tarteel_v1_confirm: 1837 / 41652
- c95 qdat: 670 / 11502
- c95 tarteel_v1_confirm: 1971 / 41572
Passed: yes
not a bar; re-graded after the one confirm read, from the committed decodes

### The pooled gating tables

**pooled confirm.1 + confirm.2** (per 1,000 graded words; n = 57935)

| cause | candidate | c95 | base | control | candidate − baseline |
|---|---|---|---|---|---|
| all_cause | 102.043 | 156.510 | 96.100 | 145.280 | -54.467 [-63.592, -47.404] |
| consonant_cause | 65.116 | 122.997 | 57.270 | 112.253 | -57.880 [-65.018, -52.166] |
| tashkeel_cause | 12.680 | 12.494 | 12.025 | 11.961 | 0.187 [-1.461, 1.527] |
| shaddah_cause | 2.850 | 2.900 | 2.652 | 2.768 | -0.050 [-0.589, 0.420] |

### Each tranche alone

**confirm.1** (per 1,000 graded words; n = 28615)

| cause | candidate | c95 | base | control | candidate − baseline |
|---|---|---|---|---|---|
| all_cause | 104.654 | 158.610 | 97.749 | 149.101 | -53.957 [-63.567, -46.161] |
| consonant_cause | 67.085 | 125.061 | 58.724 | 115.929 | -57.976 [-64.745, -51.750] |
| tashkeel_cause | 13.782 | 13.817 | 12.995 | 12.842 | -0.035 [-2.342, 1.837] |
| shaddah_cause | 2.947 | 2.869 | 2.731 | 2.917 | 0.078 [-0.825, 0.888] |
**confirm.2** (per 1,000 graded words; n = 29320)

| cause | candidate | c95 | base | control | candidate − baseline |
|---|---|---|---|---|---|
| all_cause | 99.458 | 154.459 | 94.474 | 141.506 | -55.002 [-66.190, -46.844] |
| consonant_cause | 63.168 | 120.982 | 55.835 | 108.623 | -57.814 [-67.680, -50.204] |
| tashkeel_cause | 11.590 | 11.202 | 11.069 | 11.090 | 0.388 [-1.318, 1.751] |
| shaddah_cause | 2.754 | 2.930 | 2.573 | 2.622 | -0.176 [-0.913, 0.547] |

### Gating ∧ gating.id_floor_ok

**gating.id_floor_ok** (per 1,000 graded words; n = 53161)

| cause | candidate | c95 | base | control | candidate − baseline |
|---|---|---|---|---|---|
| all_cause | 91.847 | 150.371 | 84.523 | 139.278 | -58.524 [-66.916, -52.040] |
| consonant_cause | 58.702 | 118.055 | 50.557 | 107.814 | -59.353 [-66.502, -53.205] |
| tashkeel_cause | 11.574 | 12.169 | 11.032 | 11.498 | -0.595 [-1.969, 0.556] |
| shaddah_cause | 2.302 | 2.764 | 2.228 | 2.653 | -0.462 [-0.923, 0.047] |

### Coverage and empty slots, per tranche

- confirm.1: 2835 clips, 1519 partial spans, 8 excluded
  - excluded: 106_1_1867724027.flac#0 (106:1)
  - excluded: 106_1_5166055538.flac#0 (106:1)
  - excluded: 106_1_8149234267.flac#0 (106:1)
  - excluded: 106_2_4899974765.flac#0 (106:2)
  - excluded: 106_2_6019593613.flac#0 (106:2)
  - excluded: 106_3_2657851723.flac#0 (106:3)
  - excluded: 106_4_4070109245.flac#0 (106:4)
  - excluded: 55_17_2482154935.flac#0 (55:17)
**empty slots per 100 consonant carriers** (per 1,000 or points; n = 2835)

| model | point |
|---|---|
| candidate | 16.687 |
| c95 | 19.081 |
| base | 17.138 |
| control | 18.600 |
| candidate−c95 | -2.394 [-2.877, -1.805] |
- confirm.2: 2907 clips, 1548 partial spans, 12 excluded
  - excluded: 106_1_1120059094.flac#0 (106:1)
  - excluded: 106_1_6712074310.flac#0 (106:1)
  - excluded: 106_2_5959601854.flac#0 (106:2)
  - excluded: 106_2_7809676148.flac#0 (106:2)
  - excluded: 106_3_5456853368.flac#0 (106:3)
  - excluded: 106_4_1708901254.flac#0 (106:4)
  - excluded: 106_4_3409575285.flac#0 (106:4)
  - excluded: 90_10_1410612816.flac#0 (90:10)
  - excluded: 90_10_2556015565.flac#0 (90:10)
  - excluded: 90_10_4520336054.flac#0 (90:10)
  - excluded: 90_8_3852198404.flac#0 (90:8)
  - excluded: 90_8_9835558478.flac#0 (90:8)
**empty slots per 100 consonant carriers** (per 1,000 or points; n = 2907)

| model | point |
|---|---|
| candidate | 17.460 |
| c95 | 19.480 |
| base | 17.872 |
| control | 19.041 |
| candidate−c95 | -2.020 [-2.417, -1.655] |
Graded population: 5742 clips over 113 surah clusters; 20 frozen gating clips left out (no item: the row names no ayah, or the phonetizer cannot handle the ayah the row names).
  - excluded: 106_1_1120059094.flac#0 (106:1)
  - excluded: 106_1_1867724027.flac#0 (106:1)
  - excluded: 106_1_5166055538.flac#0 (106:1)
  - excluded: 106_1_6712074310.flac#0 (106:1)
  - excluded: 106_1_8149234267.flac#0 (106:1)
  - excluded: 106_2_4899974765.flac#0 (106:2)
  - excluded: 106_2_5959601854.flac#0 (106:2)
  - excluded: 106_2_6019593613.flac#0 (106:2)
  - excluded: 106_2_7809676148.flac#0 (106:2)
  - excluded: 106_3_2657851723.flac#0 (106:3)
  - excluded: 106_3_5456853368.flac#0 (106:3)
  - excluded: 106_4_1708901254.flac#0 (106:4)
  - excluded: 106_4_3409575285.flac#0 (106:4)
  - excluded: 106_4_4070109245.flac#0 (106:4)
  - excluded: 55_17_2482154935.flac#0 (55:17)
  - excluded: 90_10_1410612816.flac#0 (90:10)
  - excluded: 90_10_2556015565.flac#0 (90:10)
  - excluded: 90_10_4520336054.flac#0 (90:10)
  - excluded: 90_8_3852198404.flac#0 (90:8)
  - excluded: 90_8_9835558478.flac#0 (90:8)

### Tarteel v1 confirm (the full round-1 reading)

**Tarteel v1 confirm** (per 1,000 graded words; n = 42300)

| cause | candidate | c95 | base | control | candidate − baseline |
|---|---|---|---|---|---|
| all_cause | 26.185 | 47.412 | 20.190 | 41.742 | -21.227 [-23.623, -18.915] |
| consonant_cause | 15.424 | 35.577 | 10.825 | 30.478 | -20.153 [-22.406, -17.924] |
| tashkeel_cause | 4.567 | 5.220 | 4.024 | 4.803 | -0.652 [-1.264, -0.052] |
| shaddah_cause | 0.311 | 0.505 | 0.311 | 0.432 | -0.194 [-0.377, -0.002] |

**Tarteel v1 confirm, word agreement with the teacher** (per 1,000 or points; n = 4408)

| model | point |
|---|---|
| candidate | 72.087 |
| c95 | 65.609 |
| candidate−c95 | 6.478 [6.046, 6.920] |

**Tarteel v1 confirm, agreement on the teacher-departure words** (per 1,000 or points; n = 4408)

| model | point |
|---|---|
| candidate | 56.155 |
| c95 | 48.485 |
| candidate−c95 | 7.670 [7.128, 8.233] |

**Tarteel v1 confirm, agreement where the teacher equals the reference** (per 1,000 or points; n = 4408)

| model | point |
|---|---|
| candidate | 92.336 |
| c95 | 87.373 |
| candidate−c95 | 4.963 [4.420, 5.529] |

### Span-trimmed sensitivity

**span-trimmed confirm.1 + confirm.2** (per 1,000 graded words; n = 57935)

| cause | candidate | c95 | base | control | candidate − baseline |
|---|---|---|---|---|---|
| all_cause | 102.043 | 156.510 | 96.100 | 145.280 | -54.467 [-63.592, -47.404] |
| consonant_cause | 65.116 | 122.997 | 57.270 | 112.253 | -57.880 [-65.018, -52.166] |
| tashkeel_cause | 12.680 | 12.494 | 12.025 | 11.961 | 0.187 [-1.461, 1.527] |
| shaddah_cause | 2.850 | 2.900 | 2.652 | 2.768 | -0.050 [-0.589, 0.420] |

### Shaddah, reported only

**shaddah agreement with the teacher** (per 1,000 or points; n = 13103)

| model | point |
|---|---|
| candidate | 95.108 |
| c95 | 89.735 |
| candidate−c95 | 5.373 [4.518, 6.416] |

### Reported companions

- ten_percent_of_c95: 15.650960469223557
- gap_closed: 0.9016369929629537
- teacher_shared_flags: {'candidate': 2995, 'c95': 2632}

### Per-reciter dose-response on Tadabur dev

one point per reciter on Tadabur dev; a per-reciter cell is descriptive and defines no interval

| reciter | clips | candidate | c95 | candidate − c95 |
|---|---|---|---|---|
| 1 | 24 | 94.769 | 95.041 | -0.271 |
| 2 | 16 | 95.014 | 95.350 | -0.336 |
| 4 | 9 | 95.871 | 96.094 | -0.223 |
| 7 | 3 | 96.947 | 97.710 | -0.763 |
| 11 | 2 | 97.196 | 97.196 | 0.000 |
| 16 | 9 | 94.056 | 94.502 | -0.446 |
| 17 | 11 | 95.916 | 96.230 | -0.314 |
| 19 | 2 | 93.506 | 94.156 | -0.649 |
| 20 | 13 | 96.429 | 97.109 | -0.680 |
| 22 | 16 | 95.464 | 95.680 | -0.216 |
| 25 | 1 | 95.327 | 96.262 | -0.935 |
| 27 | 20 | 95.614 | 96.012 | -0.399 |
| 29 | 10 | 94.575 | 95.308 | -0.733 |
| 30 | 2 | 95.402 | 96.552 | -1.149 |
| 31 | 7 | 94.706 | 95.882 | -1.176 |
| 32 | 1 | 94.836 | 96.714 | -1.878 |
| 33 | 28 | 94.196 | 94.561 | -0.366 |
| 37 | 4 | 94.332 | 93.522 | 0.810 |
| 39 | 9 | 91.322 | 92.256 | -0.935 |
| 41 | 16 | 95.335 | 95.044 | 0.292 |
| 43 | 3 | 89.868 | 90.749 | -0.881 |
| 48 | 5 | 88.387 | 87.742 | 0.645 |
| 51 | 11 | 95.466 | 95.382 | 0.084 |
| 56 | 1 | 91.026 | 88.462 | 2.564 |
| 57 | 10 | 96.021 | 95.707 | 0.314 |
| 61 | 1 | 91.275 | 91.946 | -0.671 |
| 64 | 10 | 93.249 | 93.249 | 0.000 |
| 66 | 29 | 95.477 | 95.720 | -0.244 |
| 67 | 9 | 96.471 | 97.255 | -0.784 |
| 71 | 11 | 95.765 | 95.840 | -0.074 |
| 73 | 15 | 95.390 | 96.357 | -0.967 |
| 75 | 23 | 94.880 | 94.689 | 0.191 |
| 77 | 6 | 94.677 | 93.916 | 0.760 |
| 83 | 3 | 91.915 | 91.489 | 0.426 |
| 85 | 1 | 100.000 | 100.000 | 0.000 |
| 86 | 18 | 95.738 | 95.634 | 0.104 |
| 93 | 16 | 96.376 | 96.122 | 0.254 |
| 94 | 2 | 100.000 | 100.000 | 0.000 |
| 95 | 13 | 94.512 | 94.835 | -0.323 |
| 99 | 9 | 95.401 | 94.759 | 0.642 |
| 101 | 18 | 95.596 | 95.884 | -0.289 |
| 106 | 3 | 95.575 | 96.903 | -1.327 |
| 108 | 8 | 94.135 | 93.985 | 0.150 |
| 114 | 2 | 94.624 | 94.624 | 0.000 |
| 119 | 6 | 95.200 | 95.000 | 0.200 |
| 125 | 2 | 93.227 | 94.024 | -0.797 |
| 131 | 11 | 94.897 | 95.309 | -0.412 |
| 133 | 2 | 94.420 | 94.420 | 0.000 |
| 138 | 6 | 85.560 | 85.776 | -0.216 |
| 139 | 13 | 94.667 | 94.095 | 0.571 |
| 145 | 4 | 96.380 | 96.380 | 0.000 |
| 149 | 2 | 85.185 | 87.654 | -2.469 |
| 152 | 2 | 98.540 | 99.270 | -0.730 |
| 154 | 3 | 91.525 | 91.949 | -0.424 |
| 156 | 2 | 100.000 | 100.000 | 0.000 |
| 159 | 2 | 89.790 | 91.592 | -1.802 |
| 161 | 5 | 92.453 | 92.453 | 0.000 |
| 170 | 1 | 97.087 | 97.087 | 0.000 |
| 174 | 1 | 94.937 | 92.405 | 2.532 |
| 176 | 1 | 88.462 | 88.462 | 0.000 |
| 177 | 9 | 94.712 | 94.063 | 0.649 |
| 198 | 6 | 95.126 | 95.126 | 0.000 |
| 202 | 10 | 94.424 | 94.245 | 0.180 |
| 218 | 2 | 97.015 | 96.517 | 0.498 |
| 219 | 7 | 94.557 | 94.557 | 0.000 |
| 229 | 5 | 93.643 | 94.330 | -0.687 |
| 246 | 4 | 94.966 | 95.881 | -0.915 |
| 252 | 2 | 97.902 | 97.203 | 0.699 |
| 264 | 1 | 92.537 | 94.030 | -1.493 |
| 268 | 1 | 92.593 | 92.593 | 0.000 |
| 280 | 1 | 78.947 | 78.947 | 0.000 |
| 285 | 1 | 92.857 | 91.429 | 1.429 |
| 288 | 1 | 100.000 | 100.000 | 0.000 |
| 294 | 1 | 96.552 | 96.552 | 0.000 |
| 300 | 2 | 96.000 | 96.000 | 0.000 |
| 301 | 3 | 95.455 | 95.722 | -0.267 |
| 302 | 1 | 88.889 | 88.889 | 0.000 |
| 309 | 1 | 91.045 | 91.045 | 0.000 |
| 310 | 1 | 93.985 | 95.489 | -1.504 |
| 324 | 2 | 98.095 | 98.095 | 0.000 |
| 331 | 1 | 93.103 | 93.103 | 0.000 |
| 332 | 1 | 94.030 | 94.030 | 0.000 |
| 346 | 2 | 96.250 | 95.417 | 0.833 |
| 354 | 1 | 98.387 | 98.387 | 0.000 |
| 357 | 3 | 96.212 | 96.970 | -0.758 |
| 370 | 1 | 93.305 | 94.142 | -0.837 |
| 371 | 2 | 95.238 | 96.429 | -1.190 |
| 373 | 1 | 45.455 | 59.091 | -13.636 |
| 381 | 1 | 95.614 | 94.737 | 0.877 |
| 383 | 8 | 88.745 | 89.394 | -0.649 |
| 391 | 1 | 93.333 | 90.000 | 3.333 |
| 392 | 1 | 91.250 | 91.250 | 0.000 |
| 396 | 1 | 96.875 | 96.875 | 0.000 |
| 397 | 3 | 97.163 | 95.035 | 2.128 |
| 402 | 1 | 90.826 | 91.743 | -0.917 |
| 405 | 1 | 93.571 | 94.286 | -0.714 |
| 420 | 2 | 96.797 | 97.509 | -0.712 |
| 427 | 2 | 92.920 | 92.035 | 0.885 |
| 433 | 4 | 90.683 | 90.062 | 0.621 |
| 442 | 3 | 97.239 | 97.239 | 0.000 |
| 445 | 1 | 91.515 | 92.121 | -0.606 |
| 448 | 2 | 98.462 | 95.385 | 3.077 |
| 449 | 1 | 94.872 | 94.872 | 0.000 |
| 450 | 1 | 95.023 | 95.475 | -0.452 |
| 459 | 1 | 100.000 | 100.000 | 0.000 |
| 469 | 2 | 95.683 | 97.122 | -1.439 |
| 470 | 1 | 100.000 | 97.436 | 2.564 |
| 471 | 3 | 88.742 | 89.073 | -0.331 |
| 473 | 1 | 93.458 | 93.769 | -0.312 |
| 478 | 1 | 88.710 | 88.710 | 0.000 |
| 485 | 7 | 93.392 | 93.641 | -0.249 |
| 495 | 2 | 89.600 | 90.400 | -0.800 |
| 500 | 2 | 96.117 | 97.087 | -0.971 |
| 503 | 1 | 97.727 | 98.485 | -0.758 |
| 505 | 1 | 84.444 | 82.222 | 2.222 |
| 507 | 2 | 96.916 | 96.916 | 0.000 |
| 518 | 11 | 94.510 | 94.876 | -0.366 |
| 552 | 1 | 98.113 | 98.113 | 0.000 |
| 573 | 1 | 98.148 | 100.000 | -1.852 |
| 575 | 1 | 94.737 | 94.737 | 0.000 |
| 578 | 2 | 91.519 | 90.459 | 1.060 |
| 579 | 1 | 88.889 | 88.889 | 0.000 |
| 591 | 3 | 97.727 | 97.159 | 0.568 |
| 624 | 2 | 92.381 | 94.286 | -1.905 |
| 640 | 1 | 95.588 | 95.588 | 0.000 |
| 643 | 1 | 98.333 | 98.333 | 0.000 |
| 644 | 1 | 90.210 | 90.210 | 0.000 |
| 646 | 1 | 95.000 | 95.000 | 0.000 |
| 649 | 3 | 95.055 | 95.055 | 0.000 |
| 657 | 2 | 94.118 | 93.382 | 0.735 |
| 658 | 4 | 86.312 | 87.452 | -1.141 |
| 661 | 1 | 92.391 | 94.565 | -2.174 |
| 689 | 1 | 97.674 | 93.023 | 4.651 |
| 720 | 5 | 93.761 | 94.435 | -0.675 |
| 724 | 2 | 96.319 | 95.706 | 0.613 |
| 732 | 1 | 96.154 | 96.154 | 0.000 |
| 736 | 3 | 95.596 | 95.078 | 0.518 |

### Candidate − control

n = 1 per arm: one training run each, so candidate − control is descriptive, not a test

**candidate − control** (per 1,000 or points; n = 57935)

| model | point |
|---|---|
| candidate | 102.043 |
| c95 | 156.510 |
| base | 96.100 |
| control | 145.280 |
| candidate−control | -43.237 [-52.978, -35.713] |

Reported strata: not decoded — P3 decodes the gating rows only: the reported:* strata are recorded here, never decoded, so nothing estimates them

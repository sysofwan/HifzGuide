# Shaddah probe (#86): at a geminate the decode emitted once, what did the model hear?

**Data:** [`shaddah-probe.json`](shaddah-probe.json). **Definitions, thresholds and verdict rules:**
[`shaddah-probe-preregistration.md`](shaddah-probe-preregistration.md), committed (872188e) before any
pool posterior was computed. **Code:** `tools/training/shaddah_probe.py` (measures, statistics, verdict),
`tools/training/shaddah_probe_run.py` (the run), `tools/training/ctc_paths.py` (CTC forward and Viterbi),
`tools/tadabur/time_stretch.py` (the stretch edit), `Decoder.span_log_posteriors` in
`tools/training/decoding.py`.

Re-run (GPU box, from `tools/`, ~50 min with the GPU to itself; the work directory is bound to the run's
inputs, `run_identity` in the JSON, and a directory made from other inputs is refused; this JSON comes
from a fresh identity-bound run, which reproduced every number of the first run exactly):

```bash
flock /root/scratch/gpu.lock python -m training.shaddah_probe_run \
    --audio-dir /root/scratch/issue-83/stage/clips \
    --h448 /root/repos/HifzGuide/tools/runs/h448_stream/checkpoint.pt \
    --work-dir /root/scratch/issue-86/run2 --out ../docs/shaddah-probe.json
```

Every shaddah rule in `acceptance-rules.md` stays provisional until #92 freezes the held / not held /
unsure table. Nothing here is a threshold for that table. "Held" below means held **as the model's own
output reads it**: no human verdict exists at these sites.

## Setup

- **Population:** all 3,785 kept segments of the mining pool, which come from **2,391 clips of 393
  reciters** (the manifest holds 2,508 clips of 394 reciters; the other 117 clips, and one reciter's
  only clips, have no kept segment). h448-unseen shards; every clip checksum-verified against the
  staging registry. Both models decoded each segment
  whole, bf16 weights, batch size 1, CUDA autocast (same fingerprint). The base pass reproduced the
  committed `base_decodes.json` on **all 3,785 segments** (0 mismatches).
- **Sites:** 9,430 geminate runs and 86,398 single consonants. Weighted shares use 1 / the clip's
  inclusion probability; intervals are reciter-clustered (B = 10,000, seed 20261008).

| | base teacher | h448 |
|---|---|---|
| geminates decoded **once** (collapsed) | 419 (180 reciters) | 1,157 (286 reciters) |
| of these, also in the #83 census (`contrast_sites`) | 210 | 593 |
| geminates of two decoded double | 5,705 | 5,151 |
| singles decoded once (baseline) | 83,542 | 80,324 |
| singles decoded double | 32 | 29 |

The #83 census counts 235 base sites on a normalized alignment; this probe counts collapsed **runs** on
the raw strings, which also includes ghunna runs (`مممم`, `ننن`) the census merges away. The cross-tab is
in the JSON (`geminates_by_decode_and_census`).

## Q1: the second consonant's posterior mass

`log_ratio` = log P(transcript with the consonant doubled) − log P(single), the rest of the decode fixed.
Pre-registered: **present** ≥ ln 0.1, **absent** < ln 0.001, **weak** between.

| weighted share (unweighted) | base: collapsed | base: singles | h448: collapsed | h448: singles |
|---|---|---|---|---|
| present | **53.9%** (33.7%), CI 43.2-63.7 | 0.3% | **65.5%** (57.9%), CI 59.5-71.7 | 1.3% |
| weak | 13.6% | 11.3% | 28.8% | 31.7% |
| absent | 32.4% | 88.5% | 5.6% | 67.1% |
| median `log_ratio` | −1.29 | −11.9 | −1.17 | −8.5 |
| second spike ≥ 0.1 | 25.7% | 13.7% | 63.4% | 16.7% |

The full curve (share with `log_ratio ≥ θ`):

| θ (nats) | −10 | −6.91 | −5 | −4 | −3 | −2.30 | −2 | −1 | 0 | 1 |
|---|---|---|---|---|---|---|---|---|---|---|
| base collapsed | 95.2 | 67.6 | 62.6 | 59.8 | 56.9 | 53.9 | 52.1 | 43.2 | 20.2 | 4.6 |
| base singles | 31.6 | 11.5 | 4.3 | 1.9 | 0.6 | 0.3 | 0.2 | 0.1 | 0.0 | 0.0 |
| h448 collapsed | 98.8 | 94.4 | 82.8 | 78.2 | 72.4 | 65.5 | 61.0 | 46.0 | 22.8 | 2.9 |
| h448 singles | 63.0 | 32.9 | 13.4 | 6.8 | 2.8 | 1.3 | 0.8 | 0.1 | 0.0 | 0.0 |

Geminates decoded double sit far above all of it (median `log_ratio` +11.1 base, +7.2 h448; ≥ 0 at
≈100% (99.996%) of base sites and ≈99.7% (99.70%) of h448's).

**Where the mass sits.** The output is peaky (runs of 1 frame, p90 2 frames), so a mid-run dip is
rare: the extra consonant's best alignment carves the run (`split`) at 3.8% (base) and 0.8% (h448) of
collapsed sites. Almost all of it is a **separate sub-argmax spike in the blank frames around the
emitted one**: `before` 87.1% / `after` 9.1% (base), `before` 68.4% / `after` 30.5% (h448). A geminate decoded
double is exactly this shape with the second spike above the blank. So the second consonant is, at most
collapsed sites, **present but out-voted by the blank**, not absent.

## Q3: held-segment durations

The interval between the flanking decoded tokens, at sites with a haraka on both sides; rate-normalized
by the segment's median single interval.

| median (p25-p75) | double geminates | collapsed geminates | singles |
|---|---|---|---|
| base, ms | 320 (280-360) | 200 (160-200), n = 59 | 160 (120-160) |
| base, normalized | 2.25 | 1.33 (1.0-2.0) | 1.0 |
| h448, ms | 320 (280-360) | 400 (320-800), n = 466 | 160 (120-160) |
| h448, normalized | 2.0 | 2.75 (2.0-5.0) | 1.0 |

Geminate-like (normalized ≥ the midpoint of the double and single medians: 1.625 base, 1.5 h448):
**base 33.9%** (CI 16.5-58.6, n = 55), **h448 94.8%** (CI 90.5-98.0, n = 463). Durations separate
double geminates from singles cleanly in both models: normalized ≥ 1.5 holds for 99.0% / 98.9% of
double geminates and 3.1% / 4.7% of singles (base / h448).

## Q2: stretching the held segment

Each collapsed site's interval stretched in place (WSOLA), next to an equal-length decoy (the unedited
segment plus as many zero samples **appended at its end**) and a matched single of the same consonant
stretched the same way (302 / 1,046 stretchable collapsed sites; 85 / 261 without a control in the same
clip). The decoy's placement is the registered one; that tail-only silence is neutral was not shown
(see below).

| share decoded double (weighted) | ×1.25 | ×1.5 | ×2.0 |
|---|---|---|---|
| base collapsed: stretched / decoy / **net** | 17.1 / 15.0 / **+2.1** (CI −8.8, 13.9) | 18.1 / 10.3 / **+7.8** (−6.7, 23.5) | 18.8 / 16.9 / **+2.0** (−9.4, 13.8) |
| base single control: net | 0.0 | +0.5 | **+6.2** (1.3, 15.3) |
| h448 collapsed: stretched / decoy / **net** | 15.1 / 5.1 / **+9.9** (4.3, 15.7) | 14.0 / 7.7 / **+6.3** (2.2, 10.7) | 11.4 / 9.5 / **+1.9** (−2.4, 6.4) |
| h448 single control: net | +0.1 | +0.4 | **+5.1** (2.4, 8.3) |

A longer hold does **not** reliably make either **existing model's decode** emit the second consonant:
the net flip is at most ~10 points (the base's intervals all include zero), it does not grow with the
stretch, and doubling a single's interval starts to make the decode double it (5-6%). Note the decoy:
appending silence alone flips **10-17%** (base) and 5-10% (h448) of collapsed sites to double. These
sites sit at the decision boundary, which is what Q1's mass says, and it also means the net depends on
how the decoy's length is supplied: the tail-silence decoy is not shown to be placement-neutral, so the
net is a property of this registered decoy, not a clean causal effect of the stretch.

## Pre-registered verdict

| | E1 geminate-like | E2 present gap | E3 stretch gap (×1.5) | **call** | candidate rule "looks viable" |
|---|---|---|---|---|---|
| base | 0.34 | 0.54 | 0.07 | **mixed** | no (singles unsure 11.3% > 10%) |
| h448 | 0.95 | 0.64 | 0.06 | **representation** | no (singles unsure 31.7% > 10%) |

## What drives it (exploratory, not pre-registered)

The collapsed population turned out to mix three things, so the JSON adds `exploratory_by_kind`
(by run length and whether a word starts inside the run). These splits were chosen after seeing the
results; read them as explanation, not as tests.

| collapsed sites | base: n, collapse rate | base: present | h448: n, collapse rate | h448: present | h448 geminate-like |
|---|---|---|---|---|---|
| geminate of two, within a word (`رَببِ`, `ءِللَ`) | 182, **1.6%** | 69.2% (56-79) | 574, **7.6%** | 62.0% (55-69) | 92.3% |
| run of 3+, within a word (ghunna: `ثُممممَ`, `ءِننننَ`) | 55, 2.2% | 78.7% | 264, 9.7% | 91.6% | 95.2% |
| run of 3+, across a word (tanween idgham: `ـتُوووَ`) | 182, 7.7% | **12.0%** | 319, 24.5% | 45.5% | 99.8% |

(Collapse rate = weighted share of that kind's geminate runs decoded once.)

1. **Hypothesis: many cross-word idgham sites are missed pauses, not shaddah failures.** 166 of the
   base's 182 are `و` runs from a tanween before `و`. In **17 of 20** sampled sites the base decode
   **looks pausal**: it renders the previous word in a pausal form (`سَرَه` for `سَرَتِن`, `اايَتَاا`
   for `اايَتَن`) and the `و` once; in the other 3 the segment's decode begins at the run and says
   nothing about the previous word. The sample, its selection (`random.Random(0)` over the 182 in file
   order) and each classification are committed in
   [`shaddah-probe-pausal-sample.json`](shaddah-probe-pausal-sample.json); they are the agent's reading of
   decode strings, with no audio listened to. If the reciters did pause there, segmentation missed the
   pause, the realized reference wrongly assumes wasl, and a single `و` with no second-consonant mass
   (12% present) is the faithful decode. Only adjudicated audio can confirm that; until then this is
   an exploratory hypothesis. These are 166 of the base teacher's 167 collapsed `و` sites (0.4% present)
   and, being 89% weight-1 census sites, the main reason its weighted and unweighted shares differ.
2. **At real geminates the base teacher hedges.** Its collapsed two-letter geminates are rare (1.6%),
   have short holds by its own timing (median 200 ms against 320 ms when decoded double, 160 ms for a
   single), and carry the doubled reading within a factor of 10 at ~69%. That is the profile of a
   borderline hold, which is what an *unsure* state is for.
3. **h448 collapses real geminates the teacher keeps.** Its collapse rate is ~5x the base's on
   two-letter geminates (7.6% vs 1.6%) and ~4x on ghunna runs, the holds it collapses are as long as the
   ones it decodes double (median 320 ms), and the second consonant's mass is present at 62-92%. The
   student hears the hold and fails to emit the second spike: a distillation-fidelity loss in the
   doubled encoding.

## Answer for #92

**Representation vs data, per model.**

- **base teacher: mixed.** Its collapsed population looks like two populations: cross-word idgham
  whose decodes look pausal (if the missed-pause hypothesis holds, **data**: the reference, not the
  model, is wrong; it needs audio adjudication), and within-word geminates with short holds and
  present mass (**borderline holds**, readable as *unsure* by a decode rule). No case where the teacher
  times a full-length hold and fails to emit it was found at scale.
- **h448: representation.** It perceives geminate-length holds and carries the second consonant's mass,
  but its greedy decode drops the second spike far more often than the teacher. A decode rule can
  recover most of it; distilling the second spike better (or a gemination class) would remove it.

**Does a decode rule look viable?** Not with the pre-registered band. The rule — *held* if the decode is
double or `log_ratio ≥ ln 0.1`; *unsure* if `ln 0.001 ≤ log_ratio < ln 0.1`; *not held* below — gives:

| state | base: double geminates | base: collapsed | base: singles | h448: double geminates | h448: collapsed | h448: singles |
|---|---|---|---|---|---|---|
| held | 100% | 53.9% | **0.3%** | 100% | 65.5% | **1.3%** |
| unsure | 0 | 13.6% | **11.3%** | 0 | 28.8% | **31.7%** |
| not held | 0 | 32.4% | 88.5% | 0 | 5.6% | 67.1% |

The *held* side works: at ln 0.1 it recovers 54% (base) / 66% (h448) of collapsed geminates while
calling 0.3% / 1.3% of singles held. The *unsure* band is too wide: ln 0.001 sends 11% (base) and 32%
(h448) of ordinary single consonants to unsure, and under ADR-0011 an unsure state is not graded. The
curve above shows the trade #92 has to make, e.g. a lower edge of −3 nats leaves 0.6% (base) / 2.8%
(h448) of singles at or above it and 56.9% / 72.4% of collapsed geminates. Expected behaviour of the rule
shape, whichever band #92 sets:

- a geminate decoded double → **held** (no change from today);
- a geminate decoded once with a sub-argmax second spike → **held** (most real two-letter geminates
  and ghunna runs, especially for h448);
- a geminate decoded once with no second-consonant mass → **not held** (dominated by cross-word idgham
  whose decodes look pausal; under the missed-pause hypothesis *not held* would be the right reading
  and the reference the thing that is wrong, which only adjudicated audio can settle);
- a single consonant → **not held**, except the few singles whose second spike is strong (0.3-1.3% held
  at ln 0.1), which would be added-shaddah claims.

The interval is a usable second signal (normalized ≥ 1.5: ~99% of double geminates, 3-5% of singles),
but it is measured with the model's own token timing; no operating point is proposed here.

**On stretched holds as synthetic edits** (ADR-0011 §3): this probe shows only how the **existing
models' decodes respond** to a stretched hold: neither flips reliably to double, and a ×2 stretch makes
them double some singles. It does **not** show that a stretched hold is unsuitable as training
supervision: a model trained on such edits may respond differently, and nothing here checks that a
stretched single still sounds single. Rejecting stretching as supervision would first need an
exploratory padding-placement sensitivity check of the decoy (silence at the end vs at the edit vs
spread) and the blind listen of the edits that #88 / #107 own.

## Caveats

- No human verdict exists at any collapsed site; "held" is the model's reading. The truth sites (#87)
  are what turns this into rates.
- Durations come from each model's own token spikes at 40 ms resolution.
- The census stratum makes collapsed sites over-represented relative to their weight; weighted and
  unweighted shares are both in the JSON, and they differ most where the cross-word idgham sites
  (weight ~1) dominate the counts.
- Exposure (acceptance-rules §6): the probe chose no value from these data, so the 393 reciters with a
  kept segment (2,391 clips) were used for measurement only. If #92 picks a band from these curves,
  they become shaddah-probe-tuning reciters.

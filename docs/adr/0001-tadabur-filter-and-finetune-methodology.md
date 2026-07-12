# Tadabur filtering & Muaalem fine-tune methodology

The Muaalem phoneme head was trained on professional reciters and is **over-strict**: it
rejects the imperfect-but-acceptable articulation ordinary users produce. Muraja compensates
today by running its scorer in `.balanced` mode — enabling **soft pairs**
(`ذ↔ز, ت↔ط, ض↔ظ, ك↔ق, س↔ص, ح↔ه`) and shaddah suppression — so the *algorithm* tolerates
what the *model* won't. We are fine-tuning the phoneme head on quality-filtered Tadabur
(amateur) audio to push that tolerance **into the model**, so Muraja can eventually default to
`.strict` and regain genuine discrimination between the confusable pairs.

## Decision

- **Filter = verbatim Muraja `.balanced` `ScoringParameters`.** The training-data filter reuses
  the exact, already-validated `.balanced` scorer (soft pairs + shaddahSuppression ON), ported
  faithfully from `Muraja/ios/HifzGuide/FollowAlong/`. We do **not** invent a bespoke matching
  score. The originally-proposed coverage-aware denominator (`score/max(len(query),len(ref))`)
  and hardcoded `0.70` threshold are **dropped**: they break parity with the ported
  `SmithWatermanTests` fixtures, correspond to none of the tuned modes, and penalize the
  pauses/omissions amateurs legitimately make.

- **We knowingly accept minority label-poison on the soft-pair positions.** At a reference
  position the filter cannot distinguish an *acceptable-imperfect* utterance the model
  mis-decodes (the gold example we want, "B") from a *genuinely-wrong* utterance the model
  correctly decodes as wrong ("C") — both are admitted by the same soft pair, both get the
  reference label. Separating B from C *is* the discrimination the model lacks, so no filter
  built on this model+scorer can do it. The whole approach rests on the assumption that **B ≫ C**
  (most recitation is correctly pronounced), and CTC is robust to minority label noise.

- **Poison-audit gate before training.** To turn the B≫C assumption into a measured fact, do a
  one-time human audit: ~30 randomly-admitted clips per contrast (the 6 soft pairs + shadda),
  labelled B vs C, plus a glance at the marginal ~0.65–0.72 band. Rule-of-three: 30-with-0-poison
  ⇒ true rate <~10%. Proceed if poison is small; if a contrast shows >~15%, apply
  reciter-reputation weighting or disable soft pairs for that pair (accepting the loss of some
  gold-B examples). Log audited clips (id, contrast, verdict) so the go/no-go is auditable.

- **Success criterion, in Muraja's own vocabulary:** the fine-tuned model lets Muraja default to
  `.strict` (or a tier tighter than `.balanced`) **without** raising false-negatives on
  acceptable recitation — not "lower aggregate PER."

## Consequences

- **Eval is two-sided and targeted, not aggregate PER.** Build, before training: a per-phoneme
  **confusion matrix** over all 6 soft pairs + shadda; a **"should-accept"** set of
  acceptable-imperfect amateur clips (measure recall gain vs base); and a **"should-reject"** set
  of genuinely-wrong substitutions (measure that discrimination is *retained*, not collapsed).
  Aggregate PER can *improve* while the target distinction collapses, so it cannot be the metric.

- **Tolerance moves into a non-tunable artifact.** The scorer is tunable per-context; the model
  is not. Once tolerance is baked into weights it applies everywhere, so over-shooting into
  "soft pairs indistinguishable" is a real regression the should-reject eval must catch.

- **Repeated-phrase poison reject (interior-insertion-run gate).** The Smith-Waterman scorer is a
  *local* aligner — its real job in Muraja is to locate a reciter on a mushaf page, so it trims
  unmatched query at the ends and, with the parity-locked affine gaps (`GAP_OPEN=-0.5`,
  `GAP_EXTEND=-0.1`), shrugs off insertions: a repeated phrase barely dents `match_ratio` because
  the score numerator hardly moves while only the denominator (query length) grows. Since madd is
  already collapsed before alignment (`normalize_phonemes`), any interior insertion run surviving
  to SW is genuine extra content — a strong mislabel signal when the ayah is already known. We add
  a **parity-safe, filter-side** reject: `longest_insertion_run(columns) >= MAX_INSERTION_RUN`
  (=5) fails the gate, leaving the Muraja-faithful scoring constants and `match_ratio` untouched.
  The manifest shows a clean natural gap (legit reads ≤3, poison at 6/9/13/13/14). Both the
  whole-clip filter (`filter.py`) and `segment_score.py` apply it; `segment_score` still keeps
  low-`match_ratio` segments for audit but drops repeated-phrase poison outright.

- **Two soft pairs are corpus-limited: the 30-per-contrast audit target is unreachable for
  `ت↔ط` and `ح↔ه`.** The audit worklist is sampled from *admitted* segments, so a contrast can
  only reach 30 if the corpus actually admits ≥30. Scaling the filter to shards 0–19
  (20,202 clips → 18,075 passers → 25,850 scored waqf segments) yields comfortable supply for
  `ذ↔ز`, `س↔ص`, `ض↔ظ`, `ق↔ك`, and shadda (all hit the 30 cap), but only **10** `ت↔ط` and **4**
  `ح↔ه` segments carry the contrast at all — these phoneme pairs are simply infrequent in the
  Tadabur reference distribution, and admitting more would require most of the remaining ~365
  shards for a handful of extra clips. We therefore **audit `ت↔ط` and `ح↔ه` at their full
  available supply rather than blocking the go/no-go on an unreachable n=30**. The rule-of-three
  loses power at small n (10-with-0-poison ⇒ true rate <~26%; 4-with-0-poison is only
  suggestive), so these two pairs give a weaker per-pair poison bound; if either shows *any*
  poison, treat it as a signal to disable that soft pair for training rather than trusting the
  thin sample. The over-long-clip skip in `filter.py` (`MAX_AYAH_DURATION_S`) does not affect
  this: dropped clips are pathologically mis-segmented whole-page recordings, not carriers of
  these rare contrasts.

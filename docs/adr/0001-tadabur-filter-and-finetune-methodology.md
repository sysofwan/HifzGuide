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

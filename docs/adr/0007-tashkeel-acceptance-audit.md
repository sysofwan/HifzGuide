# Measuring over-strictness: the mined tashkeel acceptance audit

ADR-0003's stated goal for the tashkeel fine-tune is two-sided: raise the model's willingness
to admit an amateur's short vowels **without** collapsing its ability to flag a genuinely wrong
one. Only the second half was ever gated. `training.counterfactual_eval` (and ADR-0006's rule
for it) measures the *cost* — errors the fine-tune stops flagging — against a fixed
`MAX_REGRESSION` margin, with no gain term in the equation at all. A fine-tune that does
exactly what it was built to do therefore reads as a failure.

The base Muaalem checkpoint was trained on professional reciters and is over-strict on
novices: on the held-out Tadabur windows it renders **84.1%** of reference vowels correctly
against the fine-tune's **98.2%**, omitting 4,659 vowels where `rung3_v2` omits 357. That gap
is the reason the fine-tune exists, and nothing in the sign-off path could see it.

## Why the existing mining pipeline could not supply the missing number

The P3.5 poison audit built `should_accept` / `should_reject` by mining the corpus with
`tadabur.contrast_attribution`, which runs `normalize_phonemes` first. That normalization
strips `U+064E`/`U+064F`/`U+0650` **unconditionally** — `مَالِكِ` becomes `مالك` — so the
scorer never flags a tashkeel disagreement and nothing tashkeel-shaped ever lands on a
worklist. The fixture sets carry buckets for `ذ↔ز`, `س↔ص`, `ض↔ظ`, `ق↔ك`, `ت↔ط`, `ح↔ه`, shadda
and `marginal`, and **no vowel bucket**. That is a structural blindness, not an oversight, and
adding a tashkeel bucket to those files would be worse than useless: the harness that reads
them back scores through the same vowel-blind scorer.

The counterfactual corpus's `control` takes *do* measure this directly, and are saturated —
**47/47 for base, `rung1_v3` and `rung3_v2` alike**. One skilled reciter articulating one
target word clearly has no discriminating power for over-strictness.

## Decision

Mine the acceptance set from Tadabur with the **vowel-aware** pass instead, and adjudicate it
by ear.

- `training.tashkeel_eval.vowel_sites` exposes the per-position provenance behind
  `VowelCounts`. `score_vowels` is now a fold of it (`count_sites`), so the aggregate gate and
  the audit worklist cannot classify the same position differently.
- `training.tashkeel_worklist` decodes base and candidate on the same held-out windows, pairs
  their sites on the **reference index** (the only coordinate two independent local alignments
  share), and emits the positions where exactly one of them matched.
- `tadabur.tashkeel_audit_ui` serves those positions for adjudication;
  `tadabur.tashkeel_fixtures` stores the verdicts; `tadabur.tashkeel_acceptance` reports the
  comparison.

### Only discordant positions are mined

The comparison is McNemar-shaped: a paired difference is a function of the discordant cells
alone. Adjudicating positions both checkpoints got right would cost audit hours and move no
number. The consequence is stated rather than hidden: this audit answers "**which checkpoint
rejects correct recitation more often, and by how much**", never "how good is either one".

### Both directions are mined, and the audit is blind

`recovered` (base failed, candidate matched) and `regressed` (the reverse) are drawn into the
same file and shuffled across buckets. Mining only the flattering direction would build a set
that can only show the fine-tune winning. The API withholds `direction` and both `*_outcome`
fields, and the shuffle exists so a listener cannot infer the direction from a run of
consecutive rows either.

The question on screen is deliberately **"which short vowel do you hear"**, not "was the model
right". Tadabur has no ground truth for the vowel a reciter *produced* — the reference records
what the mushaf prescribes — so when a checkpoint declines to mark a vowel the corpus alone
cannot distinguish an over-strict model from a reciter who genuinely said something else. Only
the ear can, and only if it is not told which answer helps.

### Sampling is per (direction, colour), and the report is two layers

Fatha outnumbers kasra and damma together, and ADR-0003's collapse check is per-colour, so a
pooled draw would leave the audit unable to speak per-colour. Because the worklist is a
*sample*, the report never presents audited counts as corpus counts: each direction yields a
confirmed **share** with a Wilson interval, which is then scaled onto the population counts the
mining run recorded in its `.summary.json` sidecar. Running the comparison without that sidecar
is a hard error, not a default.

The interval on the difference combines the two directions' Wilson bounds at their worst rather
than jointly. That is cruder and wider than necessary. Given ADR-0006's history — a Wald
interval whose width collapses to zero at `b = c = 0` would have certified non-inferiority at
`n = 10` — an interval that can only ever be too cautious is the right trade.

## Consequences

- **Sites are keyed by clip/window/reference-index**, not by sampling order, so re-mining with a
  different seed, bucket size or candidate checkpoint resumes an audit already done.
- **The audit UI plays the exact window span both checkpoints decoded**, re-encoded as 16-bit
  PCM WAV in memory, not the whole clip. Grading a vowel the model never heard would answer a
  different question. A padded "with context" take exists because a 5 s window boundary can cut
  mid-word.
- **Tadabur reciters are not novices.** They are 45 filtered corpus reciters — competent but
  imperfect. If the base checkpoint over-rejects novices specifically, this measures a **lower
  bound** on that effect. It is nonetheless real audio at roughly 100× the counterfactual set's
  41 items, available without recording anything.
- **The #10 sign-off should weigh this against the counterfactual result**, not apply
  `MAX_REGRESSION` in isolation. A margin defended without a gain term cannot approve the
  behaviour the fine-tune was built to produce.

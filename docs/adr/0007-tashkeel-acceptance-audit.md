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

### The labelled set is mined without a candidate, so the listening is not pinned behind training

Mining base-against-candidate disagreements makes the worklist — and therefore every hour of
listening — a downstream dependency of whichever fine-tune is being scored. Each training run
would restart the audit.

But the verdict a listener gives is *"the reciter said fatha"*: a fact about the audio, with no
model in it. Only the **selection** was ever coupled, and it needs to be coupled to the *base*
checkpoint, which is frozen. `training.tashkeel_worklist.static_sites` therefore stratifies on
the base outcome alone — `base_failed` / `base_matched` — which can be labelled before the next
candidate is trained. `training.tashkeel_outcomes` then supplies any later checkpoint's result
at those sites by decode, and `tashkeel_acceptance.compare_static` joins the two.

This is the same shape as `should_accept.jsonl`: the base model helped *find* the clips, the
human verdict is model-free, and every rung since has been scored against it without re-auditing
anything.

The efficiency is asymmetric, and the asymmetry decides the design:

| stratum | population | contains | yield for a good candidate |
|---|---|---|---|
| `base_failed` | 7,145 (15.9%) | every recovery any candidate can make | **≥ 88%** (`7145 − 839`) |
| `base_matched` | 37,796 | every regression any candidate can make | **≤ 2.2%** (`839 / 37796`) |

So the **gain side banks almost perfectly and the cost side does not bank at all**. `base_failed`
draws 100 per colour and `base_matched` 50 — enough to keep the estimator unbiased, not enough
to measure regressions precisely. Precision on the cost side is bought two other ways: a small
per-run `discordant_sites` top-up, and ADR-0006's counterfactual gate, which exists for exactly
that direction.

Unlike the paired path this yields each checkpoint's **absolute** false-rejection rate, because
the static strata partition every reference vowel rather than keeping only the discordant cells.

### Only discordant positions are mined (the per-run top-up)

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

### Sampling is per (direction, colour), and estimation must be too

Fatha outnumbers kasra and damma together, and ADR-0003's collapse check is per-colour, so a
pooled draw would leave the audit unable to speak per-colour. But equal draws from unequal
buckets make the *estimator* the hard part: pooling a direction's verdicts and scaling the
pooled share onto the direction's total weights each colour by its **sample** size rather than
its **population** size. With 10,000 fatha recoveries confirmed at 100% and 100 kasra
recoveries confirmed at 0%, fifty draws from each give a pooled 50% and an estimate of 5,050
confirmed sites against a true 10,000. Each stratum is therefore estimated against its own
population count, which the mining run records per `(direction, colour)` in its
`.summary.json` sidecar. Running the comparison without that sidecar is a hard error.

`unclear` verdicts leave the denominator rather than counting against confirmation: a
recording nobody can make out is not evidence that the reciter said the wrong vowel. The
report carries `unclear_share` so the weight resting on that assumption stays visible.

### The interval is simultaneous, not a difference of two 95% intervals

Differencing two independent 95% bounds yields at most 90.25% joint coverage; across six strata
the naive construction leaves roughly 74%. Labelling that `ci95` would be *anti*-conservative —
the opposite of the intended caution, and precisely the failure mode ADR-0006 already found
once, where a Wald interval whose width collapses to zero at `b = c = 0` would have certified
non-inferiority at `n = 10`. Every stratum bound entering the difference is therefore widened
by Bonferroni to `alpha / (2 * strata)`, and the report names the method and the `z` it used.

## Consequences

- **No result is shown while grading.** Exposing the running recovered/regressed tallies
  defeats the blinding on its own: verdicts are replaceable and the UI navigates backwards, so
  a listener could submit anything, watch which tally moved, and revise it knowing which answer
  flatters the fine-tune. `tadabur.tashkeel_acceptance` reports after the fact.
- **Sites are keyed by clip/window/reference-index**, not by sampling order, so re-mining with a
  different seed, bucket size or candidate checkpoint resumes an audit already done. For that to
  be worth anything the draw must also be stable under a changing population: buckets are ranked
  by a hash of the site id rather than sampled positionally, because two positional draws of 50
  from ~4,300 intersect in about 3 sites even when the populations are 95% identical.
- **The cost side is deliberately under-powered here.** `base_matched` is 84% of the corpus with
  a ~2% regression rate, so a candidate's estimate in that stratum extrapolates hard from few
  sites and carries a wide interval. That is honest rather than broken — the interval shows it —
  but a regression claim should be read off ADR-0006's counterfactual gate, not off this.
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

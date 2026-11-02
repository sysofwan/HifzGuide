# tlog mining: confirm.1 and confirm.2 open once their claims and round 2's verdict are committed (2026-11-02)

**Status:** dated 2026-11-02, frozen the day it lands, and **before any mining decode or
listening on confirm.1 or confirm.2**. This is an amendment, not an edit. The frozen
[`docs/tlog-holdout-preregistration.md`](tlog-holdout-preregistration.md) (SHA-256
`204fe7f980488aff73402c56f1729f7f17f7c9471ea0664f6365bb8870b62086`) and the dated
[`docs/tlog-gating-amendment.md`](tlog-gating-amendment.md) (SHA-256
`23d6bf7abc9882909e9686d19b2bdcd8b19fb19ed49a38829c4db9e58a4ce085`) stay byte-identical. This
document supersedes the frozen text's tranche-use rule (item 7) for the tranches §4 names, and
changes nothing else: no threshold, no population and no role assignment moves.

Issue: [#262](https://github.com/sysofwan/HifzGuide/issues/262). Parent: #261 (epic #77; tlog
#173). The pilot this amendment opens is frozen separately in
[`docs/tlog-mining-preregistration.md`](tlog-mining-preregistration.md).

## 1. The mining rule

A confirm tranche may be **mined** — decoded by any of our models, listened to and audited at
word level — once two things are committed:

- **its round's claim.** For confirm.1, `tools/tadabur/tlog/tranches/confirm.1.claim.json`
  (SHA-256 `c110fa83a1b96c0a78cbd2100a1f5f4ba7032815d304debe33269fed33a9eb88`), whose `commit`
  field is `422fdbec2521f202e5c94798ef732e2a36bec40e` (#227, dated 2026-10-31). For confirm.2,
  `tools/tadabur/tlog/tranches/confirm.2.claim.json` (SHA-256
  `1f70045452aded665adc48090c2b3b45b0b516104a669b155be62079402e437f`), with the same `commit`
  `422fdbec2521f202e5c94798ef732e2a36bec40e`. A claim's `commit` names the existing input commit
  the round opened from — the frozen manifest and its pre-registration — never its own commit.
- **its round's verdict.** For round 2, `tools/tadabur/consonant2_readout/verdict.json`
  (SHA-256 `53e7e88eabd0a5ea17ed40dd71b645935362ed9bcd8350fb189458489456555c`), committed with
  both claims at `7d8320c` ("Round 2 confirm: claims, pooled decodes, verdict FAIL on G4
  (#230) (#243)"). Its `verdict` is `fail` on G4's tashkeel cause and its `preclaim` is `pass`;
  the rule is that the verdict is committed, not that it passed. A failed round is exactly the
  case mining exists for: its read is closed, so its rows can be learned from.

Both halves are committed today. The mining rule guards the order, not the outcome.

## 2. What mining covers

The rule covers the tranche's **`reported:*` rows as well as its gating rows**. The gating rows
of confirm.1 and confirm.2 were decoded in round 2 (the three pilot detectors and the control);
**no student has decoded a `reported:*` row** — the only decode those rows carry is S7's
base-teacher identification pass. Mining decodes them for the first time.

Counts per tranche, from `tools/tadabur/tlog/tranches/meta.json`:

| tranche | gating | reported:straddles | reported:below_bar | reported:identified | reported:unclean |
|---|---|---|---|---|---|
| confirm.1 | 2,843 | 2,114 | 848 | 445 | 3,382 |
| confirm.2 | 2,919 | 2,157 | 875 | 383 | 3,413 |

`reported:undecided` is empty in both, as in every role (its rows are `unclean` first).

## 3. Status after mining

A mined tranche is:

- **never again a sealed tranche.** Sealing was a commitment not to look before the verdict; a
  mined tranche has been looked at, so no later round may treat it as sealed or replay it as one.
- **never part of any `tlog.training` version, including #258's v2.** The frozen pool predicate
  (`role == train`) excludes it; a mined `confirm.k` row is not a training row. Training on mined
  rows needs a further dated split ([`docs/tlog-mining-preregistration.md`](tlog-mining-preregistration.md),
  "Later training use").
- **never a gating set of a later round.** Its rows may be reported beside a later round's
  result, never counted in it. This is the same rule the gating amendment applies to a gating
  set once it is spent.

## 4. Scope

This amendment opens **confirm.1 and confirm.2**, and only them. Under the same rule
**confirm.3 and confirm.4** become mineable once **round 3**'s verdict is committed; until then
they stay sealed as the frozen text says. `look` was already usable and is unaffected.

## 5. Teacher-only timing passes

A **teacher-only timing pass** on look or on a spent tranche is **not a student read**, exactly
like S7's teacher-only identification of a sealed tranche: it decodes with the base teacher
(`obadx/muaalem-model-v3_2@01a1ef9f`) only, selects no student output and scores nothing. It
therefore needs **no claim** and no verdict, and it may run before or without this amendment —
slice 2's phase A does not wait for it.

What it produces is the teacher's forced alignment of each clip's **reference**. The word times
it places — never a mark a decode emitted — are what a snip is cut from; the frozen pilot design
pins that rule.

## 6. Audio

**No tlog audio is committed, ever.** The repository is **public**, and the frozen text's Source
section forbids committing or publishing tlog audio. What is committed instead is the exact
provenance and **SHA-256** of every clip: dataset revision
`9fe4f7e6d1b372c2c11157d6d2f0f91ca344eb51`, the row's `(split, shard_file, row)`, its
`audio_path` and its `audio_sha256` — the fields every tranche manifest already carries. A mined
row is identified by its manifest row; its audio is re-downloaded at the pinned revision when
needed. The owner declined a private NAS copy, since the audio can be re-downloaded at that
revision.

## 7. Owner answers (#262, 2026-11-02)

Recorded verbatim: **"#1 yes, #2 agreed, #3 yes, #4 yes."**

1. **Tranche:** re-role confirm.1/confirm.2 as mined — never sealed again, never trained on
   (without a further dated split), never a later round's gating set.
2. **Audio:** no tlog audio in git, because the repo is public and the frozen pre-registration
   forbids it; no private NAS copy either, because the audio can be re-downloaded at the pinned
   revision.
3. **Listening budget:** about 240 snips (~45–60 min), in the owner's free time, **after**
   #257's sitting, splittable into two sittings.
4. **Snipping in round 3:** #257 uses the snipped audio (slice 3) if slice 3 and phase A have
   merged before #257's worklist is served; otherwise #257 keeps whole clips and does not wait.
   #257's summary must record which presentation was used.

## 8. Changelog

- 2026-11-02: written and frozen; the owner's answers to #262's four questions recorded above.

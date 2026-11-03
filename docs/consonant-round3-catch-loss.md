# Round 3: the catch-loss audit (#257)

The pre-registered blind audit of what round 3's candidate stops flagging that round 2's model
caught. It is a **pre-claim gate**: bar 2 measures fewer flags, and this measures whether the
further drop is in false flags or in real mistakes. A failure ends the round before confirm.1 or
confirm.2 is claimed.

The rules are the pre-registration's (`docs/consonant-round3-preregistration.md`, "Catch-loss
audit", frozen 2026-10-28), and the round-2 audit's precedent is
`docs/consonant-round2-catch-loss.md`. This note records the tooling, the committed draw and the
owner's instructions; the sitting's result belongs here once it has happened.

## The populations

Look gating words graded, at the shipped default, by C95, the round-2 EMA (`r2`) and the chosen
round-3 weights (EMA; the live/EMA choice is commit `fd25c93f`).

* **New removals** (the gating stratum): C95 flags the word, `r2` flags it, and round 3 does
  not. Round 2's own removals — C95 flags, round 2 does not — are already audited (5 of 98
  decisive heard as real mistakes) and are in neither stratum.
* **New additions** (reported only): round 3 flags the word and neither C95 nor `r2` does.

A word any one of the three graded differently is in neither population. The draw reads the look
sitting's three-model export (`consonant3_readout/look_grades.jsonl.gz` plus
`look_grades_r2.jsonl.gz`) and refuses when the reference grades are absent, when they do not
repeat the export's baseline grades word for word, or when they do not reproduce the
pre-registration's disclosed counts.

## The tooling

`tools/tadabur/catch_loss_audit.py`, unchanged from round 2 except that `draw` and `score` take
`--spec` (the round's seed, reference model, root and bars):

* **`draw`** ranks each stratum by `sha256("round3-catch-loss-2026-11:<stratum>:<key>:<position>")`
  and takes the first 100 removed and 50 added (a smaller stratum whole): a seeded simple random
  sample without replacement that does not depend on row order. It writes, into
  `tools/tadabur/consonant3_readout/catch_loss/`:
  * `worklist.jsonl` — the page's whole view: an opaque id, the clip's provenance (split, shard,
    row, file name, `audio_sha256`, recorded span), the ayah, the 0-based Uthmani word index and
    the word's text, sorted by id;
  * `key.jsonl` — never served: stratum, sampling probability, and all three models' grades
    (outcome, causes, the consonant substitutions behind the flag);
  * `summary.json` — the seed, the choice's commit, each stratum's population and the sample
    size taken.
* **`score`** joins the UI's `verdicts.jsonl` (`correct` / `mistake` / `unclear`) to `key.jsonl`
  and writes `result.json`: among new removals with a decisive verdict, the real-mistake share is
  at most 20% (exactly 20% passes); fewer than 80 decisive removed verdicts cannot certify
  whatever the share. Additions, both strata by cause, unclear counts and the owner's harder
  pairs (س/ص/ث) are reported. It refuses an incomplete sitting and verdicts that are not the
  worklist's.

The UI is the #61 listening page unchanged: one seeded queue with the strata interleaved, the
whole clip from the box's look clip cache (`/root/scratch/tlog/r2/clips`, #212), each clip
verified against the worklist's `audio_sha256` before the server starts. The page receives the
ayah with the target word highlighted and nothing else: no stratum, model output, grade, weight
or tally. On cuda-dev, `scripts/issue-225-audit.sh` runs `draw`, `ui` and `backup` from a bundle
clone; `SPEC=consonant3` selects round 3's readout, scratch root
(`/root/scratch/catch_loss_r3`) and port (`8001`), so round 2's sitting keeps serving on `8000`,
untouched. Verdicts and the review list live beside the worklist in the scratch root, never
inside the clone the launcher rebuilds, so a relaunch keeps every answer and every "review
later" marker. No audio is committed and none leaves cuda-dev beyond playback.

## The committed draw, 2026-11-02

Drawn once from the round-3 look sitting's grades and committed at `df40628` (`#281`) before any
listening.

* **Seed:** `round3-catch-loss-2026-11`; choice commit `fd25c93f`.
* **Removed population:** 222 words, 100 sampled. The pre-registration disclosed 1,624 words
  flagged by both C95 and `r2`; the narrower three-model stratum (round 3 also not flagging) is
  the 222.
* **Added population:** 97 words, 50 sampled.
* **Worklist:** 150 items over 141 distinct clips; every clip's source bytes are already in the
  look clip cache, so nothing was prepared for playback.

## The owner's sitting

Not yet listened: 0 of 150 answered.

### The served page, 2026-11-02

`SPEC=consonant3 scripts/issue-225-audit.sh ui launch` on cuda-dev, with the backup loop
(`backup launch`) beside it. Round 2's UI keeps serving on `:8000`, untouched.

* **Page:** `http://10.0.1.195:8001/` — "0 of 150 answered".
* **Scratch root:** `/root/scratch/catch_loss_r3` (clone, `verdicts.jsonl`, `review.jsonl`,
  `backup/`); logs in its `logs/` (`ui.log`, `backup.log`), each job's exit status in
  `logs/<job>.exit`, absent while it runs.
* **Checked before hand-off:** the whole clip cache holds all 141 worklist hashes; over the LAN
  the page renders the ayah with the target word marked, `/api/audio?key=…&whole=0` returns mono
  16 kHz WAV and plays, and no verdict exists yet.
* **Stopping:** the `ui launch` output ends with `stop: kill -- -<pid>`, the job's process group
  (the runner and its server). Use it only with the owner's OK.

Do not launch `ui` or `draw` again while the page is up: a launch re-clones the branch under the
scratch root, the page's html and js are read from disk, and the running process holds the port,
so the sitting goes down rather than restarts. The open page also starts a fresh session while
the old answers stay in `verdicts.jsonl`; `verdicts.jsonl` and `review.jsonl` survive either way.

### What the owner does

* Listen blind: the page shows the ayah with the target word highlighted, no scores and no model
  output.
* Answer one question per item — **was the highlighted word recited correctly?** Yes / No /
  Unclear. The strata are shuffled together, so item order says nothing.
* Notes are welcome; "review later" queues an item without answering it, and the queue is not
  finished until every item has an answer.
* 150 items, roughly an hour with breaks. Audio plays from the box; nothing is downloaded.

### After the sitting

```
SPEC=consonant3 scripts/issue-225-audit.sh ui fetch DIR
git add tools/tadabur/consonant3_readout/catch_loss/{verdicts,review}.jsonl   # then commit
cd tools && python -m tadabur.catch_loss_audit score --spec consonant3        # writes result.json
```

Commit `verdicts.jsonl`, `review.jsonl` (if present) and `result.json` in
`tools/tadabur/consonant3_readout/catch_loss/`. The verdicts file is irreplaceable human data
(AGENTS.md); the backup job keeps a per-minute copy in `/root/scratch/catch_loss_r3/backup`.

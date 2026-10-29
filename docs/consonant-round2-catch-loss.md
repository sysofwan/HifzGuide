# Round 2 (tlog): the catch-loss audit (#225)

The pre-registered blind audit of what round 2's candidate stops flagging. It is a **pre-claim
gate**: bar 2 measures fewer flags, and this measures whether the drop is in false flags or in real
mistakes. A failure ends the round before confirm.1 or confirm.2 is claimed.

The rules are the pre-registration's (`docs/consonant-round2-preregistration.md`, "Catch-loss
audit"), frozen 2026-10-28; this note does not restate them, it records the tooling and the dry run.

## The tooling

`tools/tadabur/catch_loss_audit.py`

* **`draw`** reads the look sitting's committed per-word export (`consonant2_readout/
  look_grades.jsonl.gz`), the committed live/EMA choice and the committed look tranche manifest.
  A word is **removed** when C95 flags it and the candidate does not, **added** when the reverse;
  a word graded by one model only, or flagged by both or neither, is in neither population. The
  audit classifies on the export's `flagged` boolean: a word grade's `outcome` is
  `flagged_unattributed` (the carrier's group is unknown) or `flagged_elsewhere`, never
  `flagged`, and every one of the three is a flag
  (`training.muraja_policy.WORD_FLAGGED`). Each stratum is ranked by
  `sha256("round2-catch-loss-2026-10:<stratum>:<key>:<position>")` and the first 100 removed and
  50 added are taken (a smaller stratum whole), which is a seeded simple random sample without
  replacement that does not depend on the export's row order. It writes, into
  `tools/tadabur/consonant2_readout/catch_loss/`:
  * `worklist.jsonl` — the page's whole view: an opaque id, the clip's provenance (split, shard,
    row, file name, `audio_sha256`, recorded span), the ayah, the 0-based Uthmani word index and
    the word's text, sorted by id;
  * `key.jsonl` — never served: stratum, sampling probability, and both models' grade of the word
    (outcome, causes and the consonant substitutions behind the flag);
  * `summary.json` — the seed, the choice's commit, each stratum's population (words and causes)
    and the sample size taken.
* **`score`** joins `verdicts.jsonl` (the #61 UI's `correct` / `mistake` / `unclear`) to
  `key.jsonl` and writes `result.json`: the share of `mistake` among the removed stratum's decisive
  verdicts at most 20% (exactly 20% passes), fewer than 80 decisive removed verdicts
  `cannot_certify` whatever the share (cannot-certify wins over pass and fail), and the added
  stratum, both strata by cause, unclear counts and the owner's harder pairs (س/ص/ث) reported. It
  refuses a sitting with any worklist item unanswered.

Both refuse rather than guess: `draw` without the pinned pre-registration, without the committed
choice or export, on an export row that is not a look gating row of the tranche, over an existing
worklist, and when the look sitting removes no word at all (nothing could be certified, so no
worklist is drawn). `score` refuses an incomplete sitting or verdicts/lists that are not the
worklist's.

The UI is the #61 listening page's word question, one seeded queue with the strata interleaved,
whole clip from the box's look clip cache (`/root/scratch/tlog/r2/clips`, #212), each clip verified
against the worklist's `audio_sha256` before the server starts. The page receives the ayah with the
target word highlighted and nothing else: no stratum, model output, grade, weight or tally. On
cuda-dev, `scripts/issue-225-audit.sh` runs `draw`, `ui` and `backup` (a per-minute copy of the
verdicts) from a `git archive`-like bundle clone under `/root/scratch/catch_loss`, and `ui fetch`
pulls `verdicts.jsonl` back after every sitting.

## Dry run, 2026-10-29

An end-to-end dry run on **fake grades** — no model output was graded and no audit verdict counts.

* **Branch:** `agent/r2-audit` @ `271be6d`, staged as a bundle clone on cuda-dev
  (`/root/scratch/catch_loss/repo`). The round-2 pre-registration's pin is not on this branch, so
  `draw` refuses here by design; the twelve-item worklist was therefore built by the throwaway
  builder recorded below, and the real draw waits for the freeze merge.
* **Worklist:** 12 items, the first 12 rows of `tranches/look.jsonl` that are `gating`, name an
  ayah, and whose clip is in the cache; word 0 of each filename ayah (`101:4` first). Each row's
  `audio_sha256` is the tranche's own, so the UI's startup check ran against the real cache.
* **Served:** `python -m tadabur.tashkeel_audit_ui --catch-loss …/dry/worklist.jsonl --clip-cache
  /root/scratch/tlog/r2/clips --port 8125 --host 0.0.0.0` on cuda-dev, reached from the Mac at
  `http://10.0.1.195:8125/`.
* **Observed over the LAN:** the page loaded ("Listening session"; after the posted answer below,
  "1 of 12 answered"; the ayah with the target word highlighted; Yes / No / Unclear);
  `/api/sites` carried exactly
  `after, before, carrier, choices, heard, key, mode, note, review, surah_ayah`; Play set the audio
  source to `/api/audio?key=…&whole=0` and the element loaded the clip (duration 3.81 s,
  `paused: false`); `/api/audio` returned mono 16 kHz RIFF (95,154 bytes / 47,555 frames on another
  item). A posted answer was stored under the worklist id
  (`{"heard":"unclear","note":"dry run","site_id":"bcf4652778964959"}`).
* **Discarded:** the server was stopped and `verdicts.jsonl` deleted. Nothing was graded.

Limits of this record: the browser was headless Chromium on the Mac (no audio device, so
`currentTime` stayed 0 while the clip loaded and `play()` ran without error), not the owner's
device; and it played the dry-run worklist, not a drawn one. The real sitting — owner listening at
`10.0.1.195:8000` after the committed draw — is the audit itself, not a dry run.

The throwaway builder is recorded here so the dry run is reproducible; it is not part of the
tooling. It read the tranche, took each row's `filename_label` as the ayah, and wrote:

```python
{"id": sha256(f"dry:{split}:{shard_file}:{row}")[:16], "split": …, "shard_file": …, "row": …,
 "audio_filename": audio_path, "audio_sha256": …,
 "start_sample": 0, "end_sample": 0,  # the span is provenance; the page plays the whole clip
 "surah_ayah": filename_label, "word_index": 0, "word": uthmani_words(filename_label)[0]}
```

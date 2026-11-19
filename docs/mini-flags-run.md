# The published mini's confirm decode (run note, reporting only)

**What ran:** `sysofwan/hifzguide-muaalem-mini` at Hub revision
`a2bebf7c2e9a74d9ff5c3269594132d7caea7696` decoded on exactly the gating rows round 3 graded
(`confirm.3` + `confirm.4`, 5,670 gating clips), with round 3's own decoder protocol. The owner
asked for the measurement on 2026-11-18 so the v2 card can state v2's real false-flag change
against the version the app ships. It is reporting only: no gate reads it and the round's
verdict, readout and pre-registration are unchanged. The report is
[`docs/mini-flags-readout.md`](mini-flags-readout.md).

## Identity

| | |
| --- | --- |
| repository | `sysofwan/hifzguide-muaalem-mini` |
| Hub revision | `a2bebf7c2e9a74d9ff5c3269594132d7caea7696` |
| `model.safetensors` | 465,374,196 B, sha256 `2570fd3780a8435a11eb20e336bac29ac6f4dedd7c367d39abc20627e6784bb9` |
| reader | `training.mini_flags`, the `from_pretrained` path at that revision through the vendored modeling code |
| loader proof | 772/772 tensors equal the repository's own `trust_remote_code` class; a fixed seeded `(1,250,160)` window's logits equal it too — max abs diff `0.0` both |

The revision and the weights hash are the ones `docs/release-v1.0.7.md` (branch
`agent/r3-release`) pins for the published copy. `verify` records the proof in
`tools/tadabur/mini_flags_readout/decodes/verify.json`; every worker re-checks the pinned hash
and the loaded revision before decoding, and writes the same identity record beside its parts.

## The decode

- Box: `cuda-dev` (RTX 5060 Ti); tlog mining's `PAUSE` left in place, nothing resumed it.
- Command: `scripts/mini-flags-decode.sh launch`, detached under `/root/scratch/gpu.lock`
  (PID 60839, `flock … mini-flags-decode.sh run`), four processes, `OMP_NUM_THREADS=1`.
- Roles `confirm.3` and `confirm.4`, 807 shards (403 + 404), 5,670 gating rows; the tranches'
  committed claims (`3de49b29…`, `9b8b64ad…`) gate the read, as they gated the round.
- Protocol unchanged from the round's confirm decode: the `checks` store
  (`muraja-app-checks-v2`), bf16 weights, batch 1, CUDA, `fp32-features-v1`; fingerprints in
  `verify.json` and in the merge's `meta.confirm.json`.
- Inputs: the tlog parquet shards from round 3's read-only cache
  (`/root/scratch/issue-256/confirm/cache`); no clip was decoded twice.
- Run: started `2026-11-19T06:16Z`, workers `confirm.3` done `06:17:48Z`, `confirm.4` done
  `06:59:30Z`, merge exit 0 at `06:59:33Z` (≈43 min).
- Box logs: `/root/scratch/mini-flags/logs/` (`run.out`, `verify.log`, `merge.log`,
  `events`, `confirm.{3,4}.<worker>.log`); parts and output under `/root/scratch/mini-flags/`.
- Artifacts committed: `tools/tadabur/mini_flags_readout/decodes/`
  `tlog.confirm.mini.jsonl.gz` (5,670 rows), `meta.confirm.json` (code `2e4c5b6`, the claims),
  `verify.json`.

## Checks that the decode is the published model, not a broken load

- `verify` (run before any row): the downloaded hash is the pinned one, the loaded config
  resolves to the pinned revision, all 772 weight tensors and a fixed seeded `(1,250,160)`
  window equal the repository's own `trust_remote_code` forward, and the decoder's fingerprints
  are the round's for all three modes.
- Decode text is the proper 43-class phoneme alphabet at the base's length scale, not garbage.
- Character drift between the committed check texts, on every 12th pooled clip (471 clips,
  30,437 mini characters, `training.decode_evalset.levenshtein`): **mini−v2 13.26%**,
  **v2−base 8.57%**, **mini−base 15.73%** (44 and 71 and 29 of 471 clips byte-identical). The
  published mini is about 1.5× farther from the teacher than v2 on this hard population, which
  is the direction the flag table shows. The v2 card's 2.37% was 300 `gate_eval` dev clips, not
  this population, so the two are not comparable.
- Reproduce the drift with `python -m training.mini_flags drift`; it writes
  `tools/tadabur/mini_flags_readout/drift.json` (stride, clips, characters, edits, identical
  clips per pair), which is committed.

## Grading

The report grades the pooled population with the round's own harness pin
(`99c326f3…`) and shipped-default scoring, then refuses to write unless the v2 and base columns
reproduce round 3's committed table exactly — every count, denominator, point and interval. The
mini's column is the new measurement. No catch-loss audit was run, so the row says nothing about
whether the flags it removes are false.

Two denominator sensitivities are reported beside that primary table, not in place of it: the
same-words paired table (the words all three columns graded, `common_graded_terms`, denominator
1 per word) and the per-all-placed-words points. The columns' graded denominators differ (mini
45,096, v2 48,016, base 47,991 of 56,807 placed words) because a word the engine leaves ungraded
or never places leaves that column's denominator; the same-words row is the fair pairing.

The decode ran with `share_features=False`, while the round's confirm invocation used `True`.
With one model there is no feature pass to share, and the check features are model-independent,
so this cannot move a decode; the two invocations differ in the code path only, not in the
records they write.


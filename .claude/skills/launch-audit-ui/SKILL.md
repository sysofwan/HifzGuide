---
name: launch-audit-ui
description: Launch the Tadabur poison-audit web UI (tadabur.audit_ui) so it is reachable from other machines on the LAN. Use when the user wants to open, serve, restart, or expose the audit UI, or reports it is unreachable / ERR_CONNECTION_REFUSED from another device.
---

# Launch Audit UI for LAN Access

Serves `tadabur.audit_ui` — the manual poison-audit web UI that plays each sampled clip/segment
and records should-accept / should-reject labels — bound so **other machines on the LAN** can open
it, not just localhost.

**Why this is a skill and not a one-liner:** two mistakes make the UI silently unreachable, and both
look identical to the user (`ERR_CONNECTION_REFUSED`):

1. **Bound to localhost.** The server defaults to `--host 127.0.0.1`, which refuses every non-local
   connection. LAN access **requires `--host 0.0.0.0`**.
2. **Killed with its shell.** A server started inside a normal (sync) shell is a child of that
   shell and dies the moment the command returns — the port binds, then vanishes seconds later,
   leaving an **empty log and a dead PID**. The server **must be fully detached** (its own session)
   so it outlives the launching shell.

Get either wrong and the UI appears to "have been working before" but now refuses connections.

## Preconditions

- Conda env `hifzguide` exists (see `tools/README.md`).
- A **worklist** and its **audio** already exist (produced by `tadabur.audit_sampler` +
  `tadabur.segment_score` / `tadabur.waqf_segments`). This skill only *serves* them; it does not
  generate them. Typical location: `tools/tadabur/audit_run/`.

## Inputs to confirm

Ask the user (or infer from `tools/tadabur/audit_run/`) before launching:

- **Mode** — waqf-segment mode (`--segment-manifest segment_manifest.jsonl`) or full-ayah mode
  (`--manifest <filter-manifest>.jsonl`). Segment mode is the current default.
- **worklist** JSONL and **audio-dir** (segment mode: `segment_audio/`).
- **accept / reject** fixture paths to write labels to (segment runs use the run-local
  `should_accept.jsonl` / `should_reject.jsonl`, not the canonical fixtures).
- **port** (default `8000`).

If the worklist is stale relative to the manifest (e.g. after re-segmenting), regenerate it first
with `tadabur.audit_sampler` — do **not** serve a worklist that points at audio that no longer
matches the manifest.

## Procedure

Run every step from `tools/` with the env active:
`source ~/miniconda3/etc/profile.d/conda.sh && conda activate hifzguide`.

### 1. Free the port if something is already bound

```bash
ss -ltnp 2>/dev/null | grep ':8000'   # note the pid, if any
```

If a stale instance holds the port, stop it with its **literal numeric PID** (shell-security blocks
`kill $(cat ...)`; `pkill`/`killall` are not allowed):

```bash
kill <PID>
```

### 2. Launch detached, bound to 0.0.0.0

The server **must be started as a detached background process** — not a sync shell, not a plain
`nohup ... &` inside a sync command (that still dies with the session). When launching via the agent
`bash` tool, use `mode="async"` **and** `detach: true`. The command itself:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate hifzguide && cd /root/repos/HifzGuide/tools && \
exec python -m tadabur.audit_ui \
  --worklist tadabur/audit_run/audit_worklist.jsonl \
  --segment-manifest tadabur/audit_run/segment_manifest.jsonl \
  --audio-dir tadabur/audit_run/segment_audio \
  --accept tadabur/audit_run/should_accept.jsonl \
  --reject tadabur/audit_run/should_reject.jsonl \
  --port 8000 --host 0.0.0.0
```

For **full-ayah mode**, swap `--segment-manifest ...` for `--manifest <filter-manifest>.jsonl` and
point `--audio-dir` at that run's clip audio.

Record the PID for later restarts/stops:

```bash
echo <PID> > tadabur/audit_run/ui.pid
```

### 3. Verify it is actually up (do not skip)

Because the failure mode is "binds then dies," verify **after a short delay**:

```bash
sleep 5
ss -ltnp 2>/dev/null | grep ':8000'                                  # must show 0.0.0.0:8000
curl -s -o /dev/null -w "localhost HTTP %{http_code}\n" http://127.0.0.1:8000/
```

Then confirm it answers on the **LAN IP**, not just localhost:

```bash
hostname -I | tr ' ' '\n' | grep -v '^$'          # pick the LAN address, e.g. 10.0.1.195
curl -s -o /dev/null -w "lan HTTP %{http_code}\n" http://<LAN-IP>:8000/
```

Both must return `HTTP 200`. Give the user the exact URL: **`http://<LAN-IP>:8000/`**.

## Serving the listening UI with word snips

`tadabur.tashkeel_audit_ui --catch-loss` serves the catch-loss worklist as the listening UI's
whole-word question. By default every item plays its whole look clip; **`--snips`** makes it
play the word's excerpt instead, with the page's "Whole" button (`whole=1`) still fetching the
whole clip:

```bash
source ~/miniconda3/etc/profile.d/conda.sh && conda activate hifzguide && cd /root/repos/HifzGuide/tools && \
exec python -m tadabur.tashkeel_audit_ui \
  --catch-loss tadabur/consonant3_readout/catch_loss/worklist.jsonl \
  --clip-cache /root/scratch/tlog/r3/clips \
  --snips tadabur/consonant3_readout/catch_loss/snips.jsonl \
  --port 8000 --host 0.0.0.0
```

Build the sidecar **before any listening** and commit it beside the worklist:

```bash
python -m tadabur.word_snips build \
  --worklist tadabur/consonant3_readout/catch_loss/worklist.jsonl \
  --word-times tadabur/tlog_mining/word_times/look.jsonl.gz \
              tadabur/tlog_mining/word_times/confirm.1.jsonl.gz \
              tadabur/tlog_mining/word_times/confirm.2.jsonl.gz \
  --out tadabur/consonant3_readout/catch_loss/snips.jsonl \
  --report tadabur/consonant3_readout/catch_loss/snips.report.json
```

It prints the fallback rate — the share of items that play whole because the clip is not
aligned, sits below the `ll_per_frame` gate, names another ayah, has no such word, or gives
an excerpt shorter than the minimum. The sidecar is refused at startup if it misses a worklist
id or was built for another clip, and each verdict records whether the whole clip played.

### The S5 follow-along round (#311)

`tadabur.followalong_audit.s5_ui` serves the S5 draw — the same word question, the frozen draw's
own vocabulary and order. Its launcher stages the branch, pushes the audio, serves detached and
pulls the owner's verdicts back:

```bash
scripts/issue-311-audit.sh wavs push                  # the drawn clips' WAVs (498, ~140 MB)
scripts/issue-311-audit.sh ui launch                  # the pilot sitting on :8000, 0.0.0.0
WITH_MAIN=1 scripts/issue-311-audit.sh ui launch      # ... the main sitting after it, once read
scripts/issue-311-audit.sh ui check
scripts/issue-311-audit.sh ui fetch ~/s5-verdicts     # then commit
```

The pilot is served alone by default: the pre-registration may resize the main sitting after the
pilot is read, so the main one needs `WITH_MAIN=1`. The verdicts file lives in the scratch root
(`/root/scratch/s5_audit/verdicts.jsonl`) — the only copy between sittings — and `ui fetch` pulls
it into the path the scorer reads:

```bash
cp ~/s5-verdicts/verdicts.jsonl tools/tadabur/followalong_audit/s5_draw/verdicts.jsonl
git add tools/tadabur/followalong_audit/s5_draw/verdicts.jsonl   # the ! entry already tracks it
```

The WAVs are the Mac's own (`~/hifzguide-scratch/fa-apparm/wav/<audio_sha256>.wav`, the files the
App arm graded); the UI verifies each one against the committed
`app_arm/audio_manifest.jsonl.gz` before it serves a word, so a truncated or substituted clip
stops the launch instead of reaching the owner's ear. The launcher serves `:8000`; when something
else already answers there and `PORT` was not given, it retries once on `:8002` and prints the
port it used. An explicit `PORT=` gets no fallback — an occupied port then refuses the launch.

A scratch root holds one sitting at a time: while this root's UI is running, a second `ui launch`
refuses before it stages anything, even on a free port — the two would share one verdicts file and
each hold its own copy of it. Stop the running one first (`ui check` names its runner pid), then
launch; the S5 page and its `WITH_MAIN` mode are one sitting on one root, not two.

## Still refused from the user's machine?

Once the listener shows `0.0.0.0:8000` and the host's own `curl http://<LAN-IP>:8000/` returns 200,
the server is fine — remaining causes are network-level:

- **Firewall on this host.** `ufw status` (open with `sudo ufw allow 8000/tcp`) or check
  `iptables -L -n` for a blocking INPUT rule.
- **Different subnet.** Confirm the client shares the host's subnet (e.g. both `10.0.1.x`).
- **They were using a port-forward, not the raw IP.** If access previously went through an
  SSH / VS Code / editor port-forward (targets `127.0.0.1:8000`), that forward was tied to the old
  PID — re-establish the forward now that the port is live again.

## Stopping / restarting

Read the PID, then kill it with the **literal number** (`kill $(cat ...)` is blocked by
shell-security; `pkill`/`killall` are disallowed):

```bash
cat tadabur/audit_run/ui.pid     # e.g. 67630
kill 67630
```

To restart, stop first (step 1), then relaunch (step 2) and re-verify (step 3).

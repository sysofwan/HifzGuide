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

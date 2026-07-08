#!/usr/bin/env bash
# ralph-loop.sh — AFK dual-agent loop (Ralph pattern)
#
# Picks up `ready-for-agent` issues and runs a code→review loop:
#   1. Coding Agent (Opus 4.8) implements the fix/feature
#   2. Review Agent (GPT-5.5 + Opus 4.8) reviews the diff
#   3. Loop until both approve (max 3 cycles) or file HITL issue
#   4. On approval, commit directly to target branch (default: main)
#
# No PRs needed — agents handle the full review loop autonomously.
#
# Usage:
#   ./scripts/ralph-loop.sh              # run continuously
#   ./scripts/ralph-loop.sh --once       # process one issue then exit
#   ./scripts/ralph-loop.sh --dry-run    # show what would be picked up
#   ./scripts/ralph-loop.sh --issue 38   # work on a specific issue
#
# Requirements: gh, jq, copilot (GitHub Copilot CLI)

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$REPO_ROOT"
REPO="$(gh repo view --json nameWithOwner -q .nameWithOwner 2>/dev/null || echo "sysofwan/HifzGuide")"
STATE_DIR="${RALPH_STATE_DIR:-.ralph}"
STATE_FILE="$STATE_DIR/state.json"
LOG_DIR="$STATE_DIR/logs"
WORK_DIR="$STATE_DIR/work"
LABEL_READY="ready-for-agent"
LABEL_IN_PROGRESS="agent-in-progress"
COOLDOWN_SECONDS="${RALPH_COOLDOWN:-30}"
MAX_ITERATIONS="${RALPH_MAX_ITERATIONS:-50}"
MAX_REVIEW_CYCLES=3
MAX_TOTAL_CYCLES=10
EVALUATOR_MODEL="gpt-5.5"
COPILOT_CMD="${RALPH_COPILOT_CMD:-copilot}"
DEFAULT_TARGET_BRANCH="main"

# Agent model configuration
CODING_MODEL="claude-opus-4.8"
REVIEW_MODEL_1="gpt-5.5"
REVIEW_MODEL_2="claude-opus-4.8"
PR_MODEL="claude-sonnet-5"

# ─── Helpers ─────────────────────────────────────────────────────────────────

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" >&2; }
err() { log "ERROR: $*"; }

ensure_deps() {
  local cmd
  for cmd in gh jq "$COPILOT_CMD"; do
    if ! command -v "$cmd" &>/dev/null; then
      err "Required command not found: $cmd"
      exit 1
    fi
  done
}

ensure_clean_tree() {
  # Uncommitted changes
  if ! git diff --quiet 2>/dev/null || ! git diff --cached --quiet 2>/dev/null; then
    err "Working tree has uncommitted changes. Please commit, stash, or discard them first."
    exit 1
  fi

  # Unpushed commits on current branch
  local branch
  branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)
  local ahead
  ahead=$(git rev-list --count "origin/${branch}..${branch}" 2>/dev/null || echo "0")
  if [[ "$ahead" -gt 0 ]]; then
    err "Branch '$branch' has $ahead unpushed commit(s). Please push first."
    exit 1
  fi
}

acquire_lock() {
  local lockfile="$STATE_DIR/ralph.lock"
  if [[ -f "$lockfile" ]]; then
    local lock_pid
    lock_pid=$(cat "$lockfile" 2>/dev/null || echo "")
    if [[ -n "$lock_pid" ]] && kill -0 "$lock_pid" 2>/dev/null; then
      err "Another ralph-loop instance is running (PID $lock_pid)"
      exit 1
    fi
    log "Stale lock from PID $lock_pid — removing"
    rm -f "$lockfile"
  fi
  echo $$ > "$lockfile"
  trap 'cleanup_on_exit' EXIT
  trap 'log "Interrupted — cleaning up..."; exit 130' INT TERM
}

# Track the currently-in-progress issue for cleanup
CURRENT_ISSUE=""

cleanup_on_exit() {
  local lockfile="$STATE_DIR/ralph.lock"
  rm -f "$lockfile"

  # If we were mid-issue, restore labels so the issue isn't orphaned
  if [[ -n "$CURRENT_ISSUE" ]]; then
    log "Restoring labels on issue #$CURRENT_ISSUE"
    gh issue edit "$CURRENT_ISSUE" --repo "$REPO" \
      --remove-label "$LABEL_IN_PROGRESS" \
      --add-label "$LABEL_READY" 2>/dev/null || true
    # Reset git state
    git checkout main 2>/dev/null || true
    git branch -D "ralph/issue-${CURRENT_ISSUE}" 2>/dev/null || true
    CURRENT_ISSUE=""
  fi
}

reset_tree() {
  # Hard-reset the working tree to HEAD, removing all changes safely
  git reset --hard HEAD 2>/dev/null || true
  git clean -fd -e "$STATE_DIR/" 2>/dev/null || true
}

sync_target_branch() {
  local branch="$1"
  git fetch origin "$branch" 2>/dev/null || true
  git checkout "$branch" 2>/dev/null || return 1
  git reset --hard "origin/$branch" 2>/dev/null || true
}

init_state() {
  mkdir -p "$STATE_DIR" "$LOG_DIR" "$WORK_DIR"
  # Always start fresh — GitHub labels are the source of truth for retryability.
  # Only preserve 'completed' to avoid re-processing closed issues.
  local completed="[]"
  if [[ -f "$STATE_FILE" ]]; then
    completed=$(jq -r '.completed // []' "$STATE_FILE")
  fi
  cat > "$STATE_FILE" <<EOF
{
  "iteration": 0,
  "last_run": null,
  "last_issue": null,
  "completed": ${completed}
}
EOF
}

read_state() {
  ITERATION=$(jq -r '.iteration' "$STATE_FILE")
}

update_state() {
  local tmp="$STATE_FILE.tmp"
  jq --argjson iteration "$1" \
     --arg last_run "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
     --arg last_issue "${2:-null}" \
     '.iteration = $iteration | .last_run = $last_run | .last_issue = $last_issue' \
     "$STATE_FILE" > "$tmp" && mv "$tmp" "$STATE_FILE"
}

mark_completed() {
  local issue_num="$1"
  local tmp="$STATE_FILE.tmp"
  jq --argjson num "$issue_num" '.completed += [$num] | .completed |= unique' \
     "$STATE_FILE" > "$tmp" && mv "$tmp" "$STATE_FILE"
}

# ─── Issue Selection ─────────────────────────────────────────────────────────

fetch_ready_issues() {
  # Only fetch issues that are ready, not claimed, and not PRDs
  gh issue list --repo "$REPO" --state open --label "$LABEL_READY" \
    --json number,title,body,labels \
    --jq '[.[] | select(.labels | map(.name) | (index("'"$LABEL_IN_PROGRESS"'") | not) and (index("prd") | not)) | {number, title, body, labels: [.labels[].name]}]'
}

is_blocked() {
  local body="$1"
  # Extract only the "## Blocked by" section (up to next heading or end)
  local blocked_section
  blocked_section=$(echo "$body" | sed -n '/^## Blocked by/,/^## /{ /^## Blocked by/d; /^## /d; p; }')

  # If the section says "None - can start immediately", not blocked
  if [[ -z "$blocked_section" ]] || echo "$blocked_section" | grep -qi "None.*can start immediately"; then
    return 1  # not blocked
  fi

  # Extract issue references from the blocked-by section only
  # Handles both #43 shorthand and full GitHub URLs (.../issues/43)
  local refs
  refs=$(echo "$blocked_section" | grep -oE '(#[0-9]+|/issues/[0-9]+)' | grep -oE '[0-9]+' | sort -u || true)
  if [[ -z "$refs" ]]; then
    return 1  # no refs found, not blocked
  fi

  # Check if any blocking issues are still open
  local blocker_num
  for blocker_num in $refs; do
    # If we already completed this issue locally, treat it as resolved
    if jq -e --argjson n "$blocker_num" '.completed | index($n) != null' "$STATE_FILE" &>/dev/null; then
      continue
    fi
    local state
    state=$(gh issue view "$blocker_num" --repo "$REPO" --json state -q .state 2>/dev/null || echo "OPEN")
    if [[ "$state" == "OPEN" ]]; then
      return 0  # blocked
    fi
  done
  return 1  # all blockers resolved
}

is_already_processed() {
  local issue_num="$1"
  jq -e --argjson num "$issue_num" \
    '.completed | index($num) != null' \
    "$STATE_FILE" &>/dev/null
}

pick_next_issue() {
  local issues
  issues=$(fetch_ready_issues)
  local count
  count=$(echo "$issues" | jq 'length')

  if [[ "$count" -eq 0 ]]; then
    echo ""
    return
  fi

  local i
  for i in $(seq 0 $((count - 1))); do
    local num body
    num=$(echo "$issues" | jq -r ".[$i].number")
    body=$(echo "$issues" | jq -r ".[$i].body")

    if is_already_processed "$num"; then
      log "Skipping #$num (already processed)"
      continue
    fi

    if is_blocked "$body"; then
      log "Skipping #$num (blocked by open issue)"
      continue
    fi

    echo "$num"
    return
  done
  echo ""
}

# ─── Target Branch Detection ─────────────────────────────────────────────────

get_target_branch() {
  local body="$1"
  # Look for "Target branch: <branch>" — tolerates leading whitespace, backticks, bullets
  local branch
  branch=$(echo "$body" | grep -iE "^[\`\* -]*(target )?branch:" | head -1 | sed 's/.*[Bb]ranch: *//' | tr -d '[:space:]\`' || true)
  if [[ -n "$branch" && "$branch" != "branch-name>" && "$branch" != "<branch-name>" ]]; then
    echo "$branch"
  else
    echo "$DEFAULT_TARGET_BRANCH"
  fi
}

# ─── Coding Agent ────────────────────────────────────────────────────────────

run_coding_agent() {
  local issue_num="$1"
  local cycle="$2"
  local review_feedback="${3:-}"
  local issue_json
  issue_json=$(gh issue view "$issue_num" --repo "$REPO" --json title,body,labels,comments \
    --jq '{title, body, labels: [.labels[].name], comments: [.comments[].body]}')

  local title body
  title=$(echo "$issue_json" | jq -r '.title')
  body=$(echo "$issue_json" | jq -r '.body')

  # Extract agent brief from comments (the authoritative contract from triage)
  local agent_brief
  agent_brief=$(echo "$issue_json" | jq -r '.comments[] | select(contains("## Agent Brief") or contains("## What to build"))' 2>/dev/null | tail -1)

  local prompt_file="$WORK_DIR/issue-${issue_num}-code-prompt.md"

  {
    echo "You are the CODING AGENT working on issue #${issue_num}: ${title}"
    echo "Model: ${CODING_MODEL} | Cycle: ${cycle}/${MAX_REVIEW_CYCLES}"
    echo ""
    echo "## Issue"
    echo ""
    echo "$body"
    echo ""
    if [[ -n "$agent_brief" ]]; then
      echo "## Agent Brief (authoritative contract from triage)"
      echo ""
      echo "$agent_brief"
      echo ""
    fi
    if [[ -n "$review_feedback" ]]; then
      echo "## Review Feedback (Cycle $((cycle - 1)))"
      echo ""
      echo "The review agent found issues with your previous implementation. Address ALL of these:"
      echo ""
      echo "$review_feedback"
      echo ""
      echo "Your previous changes are already committed on this branch. Iterate on them — do NOT start from scratch."
      echo "Fix every point raised. Do not introduce new issues."
      echo ""
    fi
    # Always point to review history if it exists (covers edge cases)
    if [[ -f "$WORK_DIR/issue-${issue_num}-review-history.md" ]]; then
      echo "Review history: .ralph/work/issue-${issue_num}-review-history.md — read for full context on prior cycles."
      echo ""
    fi
    echo "## Instructions"
    echo ""
    echo "1. Read the issue carefully. Understand what needs to be built/fixed."
    echo "2. Explore the codebase to understand the current state."
    echo "3. Implement the fix/feature following these standards (read each from disk):"
    echo "   - .github/copilot-instructions.md — project conventions"
    echo "   - .github/skills/thermo-nuclear-code-quality/SKILL.md — structural quality bar (the review agent enforces this)"
    echo "4. Run the tests before you are done (this is a Python + data repo — see AGENTS.md):"
    echo "   - Activate the env: source ~/miniconda3/etc/profile.d/conda.sh && conda activate hifzguide"
    echo "   - For Linux/CUDA work (tools/tadabur, tools/training) ensure deps: pip install -r tools/requirements-train.txt (torch comes from the cu128 index — see tools/README.md)"
    echo "   - Run the test suite: python -m pytest -q (pip install pytest if missing; skip only if the area genuinely has no tests)"
    echo "5. Commit your changes with a descriptive message: git add -A && git commit -m 'fix/feat: description'"
    echo "   - Commit message should summarize what you changed and why."
    echo "   - Do NOT push. The review agent will review your commits next."
    echo ""
    echo "## Developer Notes (feedback to reviewer)"
    echo ""
    echo "After committing, write a brief notes file to: .ralph/work/issue-${issue_num}-dev-notes-cycle${cycle}.md"
    echo ""
    echo "Include ANY of the following that apply (omit sections that don't):"
    echo ""
    echo "### Surprises"
    echo "Unexpected things you discovered during implementation (pre-existing bugs, undocumented behavior, tricky edge cases)."
    echo ""
    echo "### Wrong assumptions"
    echo "Anything the issue description assumed incorrectly, or assumptions YOU made that turned out wrong."
    echo ""
    echo "### Deviations from spec"
    echo "Places where you intentionally diverged from the issue requirements, and why."
    echo ""
    echo "### Incomplete work"
    echo "Anything you could not finish this cycle and why (blocked, unclear, too risky without guidance)."
    echo ""
    echo "### Risks"
    echo "Potential regressions or behavioral changes the reviewer should pay extra attention to."
    echo ""
    echo "This file is NOT optional if any of the above apply. The review agent reads it for context."
    echo "If implementation was straightforward with no surprises, write a one-liner: 'Straightforward implementation, no surprises.'"
    echo ""
    echo "## HITL Escalation"
    echo ""
    echo "If you encounter ANY of these, STOP and write .ralph/hitl-reason.json then exit with code 2:"
    echo "- An architectural decision not covered by docs/adr/"
    echo "- A design choice with multiple valid approaches and no clear winner"
    echo "- A change that would break user-facing behavior in unclear ways"
    echo "- Missing information that only a human can provide"
    echo "- The issue is underspecified and you cannot proceed"
    echo ""
    echo "hitl-reason.json format:"
    echo "{"
    echo "  \"issue\": ${issue_num},"
    echo "  \"reason\": \"brief explanation\","
    echo "  \"questions\": [\"question 1\", \"question 2\"]"
    echo "}"
  } > "$prompt_file"

  local log_file="$LOG_DIR/issue-${issue_num}-code-cycle${cycle}-$(date +%Y%m%d-%H%M%S).log"
  log "  [CODE] Cycle $cycle — starting coding agent (${CODING_MODEL})"

  local exit_code=0
  "$COPILOT_CMD" --model "$CODING_MODEL" -p "$(cat "$prompt_file")" --allow-all > "$log_file" 2>&1 || exit_code=$?

  return "$exit_code"
}

# ─── Review Agent ────────────────────────────────────────────────────────────

run_review_agent() {
  local issue_num="$1"
  local cycle="$2"
  local target_branch="$3"
  local review_file="$WORK_DIR/issue-${issue_num}-review-cycle${cycle}.json"
  local review_history_file="$WORK_DIR/issue-${issue_num}-review-history.md"

  local issue_json
  issue_json=$(gh issue view "$issue_num" --repo "$REPO" --json title,body,comments \
    --jq '{title, body, comments: [.comments[].body]}')
  local title body
  title=$(echo "$issue_json" | jq -r '.title')
  body=$(echo "$issue_json" | jq -r '.body')

  # Extract agent brief from comments (the authoritative contract from triage)
  local agent_brief
  agent_brief=$(echo "$issue_json" | jq -r '.comments[] | select(contains("## Agent Brief") or contains("## What to build"))' 2>/dev/null | tail -1)

  # Get commit log on this branch (oldest first, with message bodies)
  local commit_log
  commit_log=$(git --no-pager log --reverse --format='%h %s' "$target_branch"..HEAD 2>/dev/null || echo "(no commits)")

  # Get full diff from target branch (cumulative changes)
  local diff
  diff=$(git --no-pager diff "$target_branch"...HEAD 2>/dev/null || echo "No diff available")
  if [[ -z "$diff" ]]; then
    diff=$(git diff --cached 2>/dev/null || git diff 2>/dev/null || echo "No diff available")
  fi

  # Load previous review history (if any)
  local review_history=""
  if [[ -f "$review_history_file" ]]; then
    review_history=$(cat "$review_history_file")
  fi

  local prompt_file="$WORK_DIR/issue-${issue_num}-review-prompt.md"
  {
    echo "You are the REVIEW AGENT. Two models review independently, feedback is merged."
    echo "Issue #${issue_num}: ${title}"
    echo "Cycle: ${cycle}/${MAX_REVIEW_CYCLES}"
    echo ""
    echo "## Issue requirements"
    echo ""
    echo "$body"
    echo ""
    if [[ -n "$agent_brief" ]]; then
      echo "## Agent Brief (authoritative contract — grade against this)"
      echo ""
      echo "$agent_brief"
      echo ""
    fi
    echo "## Branch commit history"
    echo ""
    echo "These are the commits on this branch (oldest first):"
    echo ""
    echo "$commit_log"
    echo ""
    if [[ -n "$review_history" ]]; then
      echo "## Previous review history"
      echo ""
      echo "Below are all previous review cycles. Check whether the coding agent addressed each point."
      echo ""
      echo "$review_history"
      echo ""
    fi
    # Load developer notes from the coding agent (current + prior cycles)
    local dev_notes_content=""
    local dn_file
    for dn_file in "$WORK_DIR"/issue-${issue_num}-dev-notes-cycle*.md; do
      [[ -f "$dn_file" ]] || continue
      local dn_cycle_label
      dn_cycle_label=$(basename "$dn_file" | grep -oE 'cycle[0-9]+')
      dev_notes_content+="#### ${dn_cycle_label}
$(cat "$dn_file")

"
    done
    if [[ -n "$dev_notes_content" ]]; then
      echo "## Developer Notes (from coding agent)"
      echo ""
      echo "The coding agent left these notes about implementation surprises, wrong assumptions, or risks."
      echo "Consider these when reviewing — they may explain seemingly odd choices or highlight areas needing extra scrutiny."
      echo ""
      echo "$dev_notes_content"
    fi
    echo "## Full diff (${target_branch}...HEAD)"
    echo ""
    echo "--- BEGIN DIFF ---"
    echo "$diff"
    echo "--- END DIFF ---"
    echo ""
    echo "## Review standard"
    echo ""
    echo "Apply this skill IN FULL — read it from disk before reviewing:"
    echo ""
    echo "1. .github/skills/thermo-nuclear-code-quality-review/SKILL.md — structural quality, code-judo, spaghetti detection"
    echo ""
    echo "Your review MUST meet the approval bar from both skills. Do not soften or skip any criterion."
    echo ""
    echo "## Additional correctness criteria"
    echo ""
    echo "1. Correctness: Does the code actually fix/implement what the issue asks?"
    echo "2. Tests: Are there adequate tests? Do they cover edge cases from the acceptance criteria?"
    echo "3. Conventions: Does it follow the project conventions (see .github/copilot-instructions.md)?"
    echo "4. Regressions: Could this break existing behavior?"
    echo "5. Completeness: Are all acceptance criteria from the issue addressed?"
    if [[ -n "$review_history" ]]; then
      echo "6. Addressed feedback: Did the coding agent address ALL points from previous reviews?"
    fi
    echo ""
    echo "## Output format"
    echo ""
    echo "You MUST write your verdict as JSON to: ${review_file}"
    echo ""
    echo "If APPROVED (no blocking issues, no structural regressions, no missed code-judo):"
    echo "{\"verdict\": \"approve\", \"notes\": \"optional minor observations\"}"
    echo ""
    echo "If CHANGES NEEDED (any presumptive blocker from the skill, or correctness issue):"
    echo "{\"verdict\": \"request_changes\", \"issues\": [{\"severity\": \"blocking\", \"description\": \"what is wrong and preferred remedy\"}]}"
    echo ""
    echo "If you need HUMAN INPUT:"
    echo "{\"verdict\": \"hitl\", \"reason\": \"why\", \"questions\": [\"question 1\"]}"
    echo ""
    echo "Write ONLY the JSON file. No other changes."
  } > "$prompt_file"

  # Run both review models and merge
  local log_file_1="$LOG_DIR/issue-${issue_num}-review${cycle}-${REVIEW_MODEL_1}-$(date +%Y%m%d-%H%M%S).log"
  local log_file_2="$LOG_DIR/issue-${issue_num}-review${cycle}-${REVIEW_MODEL_2}-$(date +%Y%m%d-%H%M%S).log"
  local review_file_1="$WORK_DIR/issue-${issue_num}-review-cycle${cycle}-model1.json"
  local review_file_2="$WORK_DIR/issue-${issue_num}-review-cycle${cycle}-model2.json"

  log "  [REVIEW] Cycle $cycle — running ${REVIEW_MODEL_1} + ${REVIEW_MODEL_2}"

  # Run both reviewers sequentially — each writes to its own output file
  local prompt_file_1="$WORK_DIR/issue-${issue_num}-review-prompt-model1.md"
  local prompt_file_2="$WORK_DIR/issue-${issue_num}-review-prompt-model2.md"
  sed "s|${review_file}|${review_file_1}|g" "$prompt_file" > "$prompt_file_1"
  sed "s|${review_file}|${review_file_2}|g" "$prompt_file" > "$prompt_file_2"

  local exit_1=0 exit_2=0
  "$COPILOT_CMD" --model "$REVIEW_MODEL_1" -p "$(cat "$prompt_file_1")" --allow-all > "$log_file_1" 2>&1 || exit_1=$?
  "$COPILOT_CMD" --model "$REVIEW_MODEL_2" -p "$(cat "$prompt_file_2")" --allow-all > "$log_file_2" 2>&1 || exit_2=$?

  if [[ "$exit_1" -ne 0 ]]; then
    log "  [REVIEW] Warning: ${REVIEW_MODEL_1} exited with code $exit_1"
  fi
  if [[ "$exit_2" -ne 0 ]]; then
    log "  [REVIEW] Warning: ${REVIEW_MODEL_2} exited with code $exit_2"
  fi

  # Merge review verdicts
  merge_reviews "$review_file_1" "$review_file_2" "$review_file"

  return 0
}

merge_reviews() {
  local file1="$1" file2="$2" output="$3"

  local v1 v2
  v1=$(jq -r '.verdict // "error"' "$file1" 2>/dev/null || echo "error")
  v2=$(jq -r '.verdict // "error"' "$file2" 2>/dev/null || echo "error")

  # If either says HITL, result is HITL
  if [[ "$v1" == "hitl" || "$v2" == "hitl" ]]; then
    local hitl_source="$file1"
    [[ "$v2" == "hitl" ]] && hitl_source="$file2"
    cp "$hitl_source" "$output"
    return
  fi

  # If either requests changes, merge all issues
  if [[ "$v1" == "request_changes" || "$v2" == "request_changes" ]]; then
    local issues1 issues2
    issues1=$(jq '.issues // []' "$file1" 2>/dev/null || echo "[]")
    issues2=$(jq '.issues // []' "$file2" 2>/dev/null || echo "[]")
    jq -n --argjson i1 "$issues1" --argjson i2 "$issues2" \
      '{verdict: "request_changes", issues: ($i1 + $i2)}' > "$output"
    return
  fi

  # BOTH must approve — a single approval is not sufficient
  if [[ "$v1" == "approve" && "$v2" == "approve" ]]; then
    echo '{"verdict": "approve", "notes": "Both reviewers approved."}' > "$output"
    return
  fi

  # One approved but other errored — treat as request_changes (require both)
  if [[ "$v1" == "approve" && "$v2" == "error" ]] || [[ "$v2" == "approve" && "$v1" == "error" ]]; then
    echo '{"verdict": "request_changes", "issues": [{"severity": "blocking", "description": "One review model failed to produce output. Only one approval is insufficient — both models must agree."}]}' > "$output"
    return
  fi

  # Both errored
  echo '{"verdict": "request_changes", "issues": [{"severity": "blocking", "description": "Both review agents failed to produce valid output. Retry."}]}' > "$output"
}

# ─── Evaluator Agent ─────────────────────────────────────────────────────────

run_evaluator_agent() {
  local issue_num="$1"
  local cycle="$2"
  local target_branch="$3"
  local eval_file="$WORK_DIR/issue-${issue_num}-eval-cycle${cycle}.json"

  local issue_json
  issue_json=$(gh issue view "$issue_num" --repo "$REPO" --json title,body \
    --jq '{title, body}')
  local title body
  title=$(echo "$issue_json" | jq -r '.title')
  body=$(echo "$issue_json" | jq -r '.body')

  # Get commit log and diff
  local commit_log
  commit_log=$(git --no-pager log --reverse --format='%h %s' "$target_branch"..HEAD 2>/dev/null || echo "(no commits)")
  local diff_stat
  diff_stat=$(git --no-pager diff --stat "$target_branch"...HEAD 2>/dev/null || echo "")

  # Load review history
  local review_history=""
  local review_history_file="$WORK_DIR/issue-${issue_num}-review-history.md"
  if [[ -f "$review_history_file" ]]; then
    review_history=$(cat "$review_history_file")
  fi

  local prompt_file="$WORK_DIR/issue-${issue_num}-eval-prompt.md"
  {
    echo "You are the EVALUATOR AGENT. Your job is to decide whether to CONTINUE or GIVE UP."
    echo ""
    echo "Issue #${issue_num}: ${title}"
    echo "Completed cycles: ${cycle}/${MAX_TOTAL_CYCLES} (hard limit)"
    echo ""
    echo "## Issue requirements"
    echo ""
    echo "$body"
    echo ""
    echo "## Branch commit history"
    echo ""
    echo "$commit_log"
    echo ""
    echo "## Diff stats"
    echo ""
    echo "$diff_stat"
    echo ""
    echo "## Full review history (all code→review cycles so far)"
    echo ""
    echo "$review_history"
    echo ""
    # Include developer notes for evaluator context
    local dev_notes_eval=""
    local dn_file
    for dn_file in "$WORK_DIR"/issue-${issue_num}-dev-notes-cycle*.md; do
      [[ -f "$dn_file" ]] || continue
      local dn_label
      dn_label=$(basename "$dn_file" | grep -oE 'cycle[0-9]+')
      dev_notes_eval+="#### ${dn_label}
$(cat "$dn_file")

"
    done
    if [[ -n "$dev_notes_eval" ]]; then
      echo "## Developer notes (coding agent's own observations)"
      echo ""
      echo "The coding agent reported these findings — consider whether they indicate the task is tractable or stuck."
      echo ""
      echo "$dev_notes_eval"
    fi
    echo "## Decision criteria"
    echo ""
    echo "CONTINUE if:"
    echo "- The code is making clear progress toward the goal"
    echo "- The remaining review feedback is addressable (not fundamental design disagreement)"
    echo "- The latest cycle made meaningful progress on prior feedback"
    echo "- The issue is close to being solved (e.g., minor fixes remaining)"
    echo ""
    echo "GIVE UP if:"
    echo "- The same feedback keeps repeating without progress (loop is stuck)"
    echo "- The remaining issues require fundamental architectural changes the agent cannot resolve"
    echo "- The code quality is regressing instead of improving"
    echo "- The agent is making changes unrelated to the review feedback"
    echo "- The problem is genuinely beyond automated resolution (needs human insight)"
    echo ""
    echo "## Output"
    echo ""
    echo "Write your decision as JSON to: ${eval_file}"
    echo ""
    echo "If CONTINUE:"
    echo "{\"decision\": \"continue\", \"reason\": \"why you believe more cycles will succeed\"}"
    echo ""
    echo "If GIVE UP:"
    echo "{\"decision\": \"give_up\", \"reason\": \"why continuing would be wasteful\", \"summary\": \"what was accomplished and what remains\"}"
    echo ""
    echo "Write ONLY the JSON file. No other changes."
  } > "$prompt_file"

  local log_file="$LOG_DIR/issue-${issue_num}-eval-cycle${cycle}-$(date +%Y%m%d-%H%M%S).log"
  log "  [EVAL] Running evaluator (${EVALUATOR_MODEL})"

  "$COPILOT_CMD" --model "$EVALUATOR_MODEL" -p "$(cat "$prompt_file")" --allow-all > "$log_file" 2>&1 || true

  # Parse decision
  if [[ ! -f "$eval_file" ]]; then
    log "  [EVAL] No output — defaulting to give_up"
    echo '{"decision": "give_up", "reason": "Evaluator agent failed to produce output"}' > "$eval_file"
  fi

  local decision
  decision=$(jq -r '.decision // "give_up"' "$eval_file" 2>/dev/null || echo "give_up")
  echo "$decision"
}

# ─── Code→Review Loop ────────────────────────────────────────────────────────

run_code_review_loop() {
  local issue_num="$1"
  local target_branch="$2"
  local review_feedback=""

  # Sync target branch and create work branch from latest remote
  sync_target_branch "$target_branch" || {
    err "Could not sync target branch $target_branch"
    return 1
  }
  local work_branch="ralph/issue-${issue_num}"
  git checkout -B "$work_branch" "$target_branch" 2>/dev/null

  # Clear stale review history from any previous run of this issue
  rm -f "$WORK_DIR/issue-${issue_num}-review-history.md"

  local cycle
  for cycle in $(seq 1 "$MAX_REVIEW_CYCLES"); do
    log "─── Issue #$issue_num — Cycle $cycle/$MAX_REVIEW_CYCLES ───"

    # 1. Run coding agent
    local code_exit=0
    run_coding_agent "$issue_num" "$cycle" "$review_feedback" || code_exit=$?

    if [[ "$code_exit" -eq 2 ]]; then
      return 2  # HITL
    elif [[ "$code_exit" -ne 0 ]]; then
      err "Coding agent crashed (exit $code_exit)"
      return 1
    fi

    # 2. Ensure we're on the work branch (coding agent might have switched)
    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    if [[ "$current_branch" != "$work_branch" ]]; then
      log "  Warning: coding agent left us on '$current_branch', restoring '$work_branch'"
      git checkout "$work_branch" 2>/dev/null || return 1
    fi

    # 3. Ensure ALL changes are committed (so review sees exactly what would land)
    git add -A 2>/dev/null || true
    if ! git diff --cached --quiet 2>/dev/null; then
      git commit -m "fix(#${issue_num}): cycle ${cycle} implementation" --no-verify 2>/dev/null || true
    fi

    # 4. Check if branch has any commits beyond target
    local has_changes=false
    if ! git diff "$target_branch"..."$work_branch" --quiet 2>/dev/null; then
      has_changes=true
    fi

    if [[ "$has_changes" == "false" ]]; then
      err "Coding agent produced no changes after cycle $cycle"
      # Check if coding agent left dev notes explaining why
      local no_change_notes="$WORK_DIR/issue-${issue_num}-dev-notes-cycle${cycle}.md"
      if [[ -f "$no_change_notes" ]]; then
        log "  [CODE] Developer notes (no-change explanation):"
        sed 's/^/    /' "$no_change_notes" >&2
        review_feedback="You produced no code changes in the last cycle. Your own notes say:
$(cat "$no_change_notes")

You MUST produce code changes. If you believe no changes are needed, explain via HITL escalation instead."
      else
        # Surface tail of agent log so we can see what happened
        local last_log
        last_log=$(ls -t "$LOG_DIR"/issue-${issue_num}-code-cycle${cycle}-*.log 2>/dev/null | head -1)
        if [[ -n "$last_log" && -f "$last_log" ]]; then
          log "  [CODE] Agent log tail (no changes produced):"
          tail -20 "$last_log" | sed 's/^/    /' >&2
        fi
        review_feedback="You produced no code changes in the last cycle. The review needs actual implementation. If you cannot implement, write dev notes to .ralph/work/issue-${issue_num}-dev-notes-cycle${cycle}.md explaining WHY, then commit at least a stub or escalate via HITL."
      fi
      continue
    fi

    # 5. Run review agent (reviews committed state — exactly what would land)
    run_review_agent "$issue_num" "$cycle" "$target_branch"

    # 4. Parse review verdict
    local review_file="$WORK_DIR/issue-${issue_num}-review-cycle${cycle}.json"
    if [[ ! -f "$review_file" ]]; then
      err "Review agent produced no output"
      review_feedback="The review agent did not produce output. Please ensure your changes are complete and correct."
      continue
    fi

    local verdict
    verdict=$(jq -r '.verdict' "$review_file" 2>/dev/null || echo "error")

    case "$verdict" in
      approve)
        log "  [REVIEW] ✓ APPROVED on cycle $cycle"
        # Append approval to history
        local review_history_file="$WORK_DIR/issue-${issue_num}-review-history.md"
        {
          echo "### Cycle $cycle — APPROVED"
          echo ""
          echo "$(jq -r '.notes // "No additional notes."' "$review_file")"
          echo ""
        } >> "$review_history_file"
        return 0
        ;;
      hitl)
        log "  [REVIEW] HITL requested"
        jq --argjson num "$issue_num" '. + {issue: $num}' "$review_file" > "$STATE_DIR/hitl-reason.json"
        return 2
        ;;
      request_changes)
        review_feedback=$(jq -r '.issues | map("- [\(.severity)] \(.description)") | join("\n")' "$review_file")
        log "  [REVIEW] ✗ Changes requested:"
        echo "$review_feedback" | sed 's/^/    /' >&2
        # Append to persistent review history file
        local review_history_file="$WORK_DIR/issue-${issue_num}-review-history.md"
        {
          echo "### Cycle $cycle — CHANGES REQUESTED"
          echo ""
          echo "$review_feedback"
          echo ""
        } >> "$review_history_file"
        ;;
      *)
        err "Invalid review verdict: $verdict"
        review_feedback="Previous review produced an invalid verdict. Please ensure your implementation is complete."
        ;;
    esac
  done

  # Exhausted initial review cycles — run evaluator to decide continue vs give up
  local total_cycles="$MAX_REVIEW_CYCLES"

  while [[ "$total_cycles" -lt "$MAX_TOTAL_CYCLES" ]]; do
    log "  [EVAL] Cycles exhausted ($total_cycles/$MAX_TOTAL_CYCLES) — consulting evaluator"

    local eval_decision
    eval_decision=$(run_evaluator_agent "$issue_num" "$total_cycles" "$target_branch")

    if [[ "$eval_decision" != "continue" ]]; then
      local eval_file="$WORK_DIR/issue-${issue_num}-eval-cycle${total_cycles}.json"
      local give_up_reason
      give_up_reason=$(jq -r '.reason // "No reason given"' "$eval_file" 2>/dev/null)
      local give_up_summary
      give_up_summary=$(jq -r '.summary // ""' "$eval_file" 2>/dev/null)

      log "  [EVAL] Decision: GIVE UP — $give_up_reason"

      # Comment on the issue explaining why
      gh issue comment "$issue_num" --repo "$REPO" --body \
        "> *This was generated by AI during triage.*

## Evaluator Decision: Give Up (after $total_cycles cycles)

**Reason:** $give_up_reason

**Summary:** $give_up_summary

The work branch \`ralph/issue-${issue_num}\` is preserved for manual review." 2>/dev/null || true

      return 1
    fi

    log "  [EVAL] Decision: CONTINUE"
    total_cycles=$((total_cycles + 1))

    # Run one more code→review cycle
    log "─── Issue #$issue_num — Cycle $total_cycles/$MAX_TOTAL_CYCLES (extended) ───"

    local code_exit=0
    run_coding_agent "$issue_num" "$total_cycles" "$review_feedback" || code_exit=$?

    if [[ "$code_exit" -eq 2 ]]; then
      return 2
    elif [[ "$code_exit" -ne 0 ]]; then
      err "Coding agent crashed (exit $code_exit)"
      return 1
    fi

    # Ensure work branch and commit
    local current_branch
    current_branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "")
    if [[ "$current_branch" != "$work_branch" ]]; then
      git checkout "$work_branch" 2>/dev/null || return 1
    fi
    git add -A 2>/dev/null || true
    if ! git diff --cached --quiet 2>/dev/null; then
      git commit -m "fix(#${issue_num}): cycle ${total_cycles} implementation" --no-verify 2>/dev/null || true
    fi

    if git diff "$target_branch"..."$work_branch" --quiet 2>/dev/null; then
      err "Coding agent produced no changes after cycle $total_cycles"
      local no_change_notes="$WORK_DIR/issue-${issue_num}-dev-notes-cycle${total_cycles}.md"
      if [[ -f "$no_change_notes" ]]; then
        log "  [CODE] Developer notes (no-change explanation):"
        sed 's/^/    /' "$no_change_notes" >&2
        review_feedback="You produced no code changes in the last cycle. Your own notes say:
$(cat "$no_change_notes")

You MUST produce code changes or escalate via HITL."
      else
        local last_log
        last_log=$(ls -t "$LOG_DIR"/issue-${issue_num}-code-cycle${total_cycles}-*.log 2>/dev/null | head -1)
        if [[ -n "$last_log" && -f "$last_log" ]]; then
          log "  [CODE] Agent log tail (no changes produced):"
          tail -20 "$last_log" | sed 's/^/    /' >&2
        fi
        review_feedback="You produced no code changes. Write dev notes explaining why, then implement or escalate via HITL."
      fi
      continue
    fi

    run_review_agent "$issue_num" "$total_cycles" "$target_branch"

    local review_file="$WORK_DIR/issue-${issue_num}-review-cycle${total_cycles}.json"
    if [[ ! -f "$review_file" ]]; then
      review_feedback="The review agent did not produce output."
      continue
    fi

    local verdict
    verdict=$(jq -r '.verdict' "$review_file" 2>/dev/null || echo "error")

    case "$verdict" in
      approve)
        log "  [REVIEW] ✓ APPROVED on cycle $total_cycles"
        local review_history_file="$WORK_DIR/issue-${issue_num}-review-history.md"
        {
          echo "### Cycle $total_cycles — APPROVED"
          echo ""
          echo "$(jq -r '.notes // "No additional notes."' "$review_file")"
          echo ""
        } >> "$review_history_file"
        return 0
        ;;
      hitl)
        jq --argjson num "$issue_num" '. + {issue: $num}' "$review_file" > "$STATE_DIR/hitl-reason.json"
        return 2
        ;;
      request_changes)
        review_feedback=$(jq -r '.issues | map("- [\(.severity)] \(.description)") | join("\n")' "$review_file")
        log "  [REVIEW] ✗ Changes requested:"
        echo "$review_feedback" | sed 's/^/    /' >&2
        local review_history_file="$WORK_DIR/issue-${issue_num}-review-history.md"
        {
          echo "### Cycle $total_cycles — CHANGES REQUESTED"
          echo ""
          echo "$review_feedback"
          echo ""
        } >> "$review_history_file"
        ;;
      *)
        review_feedback="Previous review produced an invalid verdict."
        ;;
    esac
  done

  # Hard limit reached
  log "  Hard limit of $MAX_TOTAL_CYCLES cycles reached"
  gh issue comment "$issue_num" --repo "$REPO" --body \
    "> *This was generated by AI during triage.*

## Hard Limit Reached ($MAX_TOTAL_CYCLES cycles)

The AFK agent exhausted all available cycles. Work branch \`ralph/issue-${issue_num}\` preserved." 2>/dev/null || true
  return 1
}

# ─── Create PR & Auto-Merge ──────────────────────────────────────────────────

create_pr_summary() {
  local issue_num="$1"
  local target_branch="$2"
  local work_branch="ralph/issue-${issue_num}"
  local pr_body_file="$WORK_DIR/issue-${issue_num}-pr-body.md"

  # Gather context for the PR agent
  local issue_json
  issue_json=$(gh issue view "$issue_num" --repo "$REPO" --json title,body \
    --jq '{title, body}')
  local title
  title=$(echo "$issue_json" | jq -r '.title')

  local commit_log
  commit_log=$(git --no-pager log --reverse --format='%h %s' "$target_branch".."$work_branch" 2>/dev/null || echo "(no commits)")

  local full_diff
  full_diff=$(git --no-pager diff "$target_branch"..."$work_branch" 2>/dev/null || echo "")

  local review_history=""
  local review_history_file="$WORK_DIR/issue-${issue_num}-review-history.md"
  if [[ -f "$review_history_file" ]]; then
    review_history=$(cat "$review_history_file")
  fi

  local prompt_file="$WORK_DIR/issue-${issue_num}-pr-prompt.md"
  {
    echo "You are a PR SUMMARY AGENT. Write a pull request description for the work done on issue #${issue_num}."
    echo ""
    echo "Write the PR body to: ${pr_body_file}"
    echo ""
    echo "## Issue"
    echo ""
    echo "Title: ${title}"
    echo ""
    echo "## Commit history"
    echo ""
    echo "\`\`\`"
    echo "$commit_log"
    echo "\`\`\`"
    echo ""
    echo "## Full diff"
    echo ""
    echo "\`\`\`diff"
    echo "$full_diff"
    echo "\`\`\`"
    echo ""
    echo "## Review history (code→review cycles)"
    echo ""
    echo "$review_history"
    echo ""
    # Include developer notes
    local dev_notes_all=""
    local dn_file
    for dn_file in "$WORK_DIR"/issue-${issue_num}-dev-notes-cycle*.md; do
      [[ -f "$dn_file" ]] || continue
      local dn_label
      dn_label=$(basename "$dn_file" | grep -oE 'cycle[0-9]+')
      dev_notes_all+="#### ${dn_label}
$(cat "$dn_file")

"
    done
    if [[ -n "$dev_notes_all" ]]; then
      echo "## Developer notes (from coding agent)"
      echo ""
      echo "$dev_notes_all"
    fi
    echo "## PR body format"
    echo ""
    echo "Write the PR body in this structure (markdown):"
    echo ""
    echo "### Summary"
    echo "What was implemented and why (2-3 sentences connecting to the issue)."
    echo ""
    echo "### Implementation"
    echo "Key changes made — files touched, patterns used, design choices."
    echo ""
    echo "### Surprising findings"
    echo "Anything unexpected discovered during implementation — edge cases, pre-existing issues, tricky parts. Omit this section if nothing surprising."
    echo ""
    echo "### Reviewer feedback incorporated"
    echo "Summarize the review cycles — what was flagged and how it was addressed. Omit if approved on first cycle."
    echo ""
    echo "### Architectural changes"
    echo "Any structural/design shifts from the original plan. Omit if implementation was straightforward."
    echo ""
    echo "### Testing"
    echo "How the changes were validated (tests added/modified, manual verification)."
    echo ""
    echo "---"
    echo ""
    echo "Keep it concise but informative. Write ONLY the file, no other changes."
  } > "$prompt_file"

  local log_file="$LOG_DIR/issue-${issue_num}-pr-summary-$(date +%Y%m%d-%H%M%S).log"
  log "  [PR] Generating PR summary (${PR_MODEL})"

  "$COPILOT_CMD" --model "$PR_MODEL" -p "$(cat "$prompt_file")" --allow-all > "$log_file" 2>&1 || true

  # Fallback if agent didn't produce the file
  if [[ ! -f "$pr_body_file" ]] || [[ ! -s "$pr_body_file" ]]; then
    log "  [PR] Agent didn't produce summary — using fallback"
    {
      echo "## Summary"
      echo ""
      echo "Implements #${issue_num}."
      echo ""
      echo "## Commits"
      echo ""
      echo "\`\`\`"
      echo "$commit_log"
      echo "\`\`\`"
      echo ""
      if [[ -n "$review_history" ]]; then
        echo "## Review history"
        echo ""
        echo "$review_history"
      fi
    } > "$pr_body_file"
  fi
}

commit_and_land() {
  local issue_num="$1"
  local target_branch="$2"
  local work_branch="ralph/issue-${issue_num}"

  # Safety: verify no unreviewed changes slipped in after approval
  git add -A 2>/dev/null || true
  if ! git diff --cached --quiet 2>/dev/null; then
    err "Unreviewed changes detected after approval — aborting land"
    return 1
  fi

  # Push work branch to remote
  log "  [PR] Pushing work branch to origin"
  if ! git push -u origin "$work_branch" --force-with-lease 2>&1; then
    err "Failed to push work branch $work_branch"
    return 1
  fi

  # Generate PR summary
  create_pr_summary "$issue_num" "$target_branch"

  local pr_body_file="$WORK_DIR/issue-${issue_num}-pr-body.md"
  local issue_title
  issue_title=$(gh issue view "$issue_num" --repo "$REPO" --json title -q .title 2>/dev/null || echo "Issue #${issue_num}")

  # Create the PR
  log "  [PR] Creating pull request"
  local pr_url
  pr_url=$(gh pr create --repo "$REPO" \
    --base "$target_branch" \
    --head "$work_branch" \
    --title "fix(#${issue_num}): ${issue_title}" \
    --body-file "$pr_body_file" \
    2>&1) || {
    err "Failed to create PR: $pr_url"
    return 1
  }

  local pr_num
  pr_num=$(echo "$pr_url" | grep -oE '[0-9]+$')
  log "  [PR] Created PR #$pr_num → $pr_url"

  # Wait for PR checks to pass before merging
  log "  [PR] Waiting for status checks to complete..."
  local check_result
  local max_wait=1800  # 30 minutes max
  local waited=0
  local poll_interval=15

  while [[ "$waited" -lt "$max_wait" ]]; do
    check_result=$(gh pr checks "$pr_num" --repo "$REPO" --json name,bucket \
      --jq '[.[] | .bucket] | if length == 0 then "none" elif all(. == "pass") then "done" elif any(. == "fail") then "failed" else "pending" end' 2>/dev/null || echo "none")

    case "$check_result" in
      done)
        log "  [PR] All checks passed"
        break
        ;;
      failed)
        err "PR #$pr_num checks failed"
        return 1
        ;;
      none)
        log "  [PR] No status checks configured — proceeding"
        break
        ;;
      *)
        if [[ "$((waited % 60))" -eq 0 ]] && [[ "$waited" -gt 0 ]]; then
          log "  [PR] Checks still pending (${waited}s elapsed)..."
        fi
        sleep "$poll_interval"
        waited=$((waited + poll_interval))
        ;;
    esac
  done

  if [[ "$waited" -ge "$max_wait" ]]; then
    err "PR #$pr_num checks timed out after ${max_wait}s"
    return 1
  fi

  # Merge the PR
  log "  [PR] Merging PR #$pr_num"
  if ! gh pr merge "$pr_num" --repo "$REPO" --merge 2>&1; then
    err "Failed to merge PR #$pr_num"
    return 1
  fi

  # Clean up local work branch
  git checkout "$target_branch" 2>/dev/null || true
  git pull origin "$target_branch" 2>/dev/null || true
  git branch -D "$work_branch" 2>/dev/null || true

  log "  [PR] PR #$pr_num merged into $target_branch"
  return 0
}

# ─── HITL Filing ─────────────────────────────────────────────────────────────

file_hitl_issue() {
  local issue_num="$1"
  local reason_file="$STATE_DIR/hitl-reason.json"

  if [[ ! -f "$reason_file" ]]; then
    err "HITL signaled but no reason file found"
    return 1
  fi

  local reason questions
  reason=$(jq -r '.reason' "$reason_file")
  questions=$(jq -r '.questions | map("- " + .) | join("\n")' "$reason_file")

  local hitl_body
  hitl_body=$(cat <<EOF
> *This was generated by AI during triage.*

## Context

The AFK agent (ralph-loop) was working on #${issue_num} but needs human input.

## Why this needs a human

${reason}

## Questions for the maintainer

${questions}

## Agent config

- Code model: ${CODING_MODEL}
- Review models: ${REVIEW_MODEL_1} + ${REVIEW_MODEL_2}
- Max cycles: ${MAX_REVIEW_CYCLES}

## Original issue

Linked: #${issue_num}
EOF
)

  local hitl_issue_num
  hitl_issue_num=$(gh issue create --repo "$REPO" \
    --title "HITL: Human input needed for #${issue_num}" \
    --body "$hitl_body" \
    --label "needs-triage" \
    2>&1 | grep -oE '[0-9]+$')

  gh issue comment "$issue_num" --repo "$REPO" --body \
    "> *This was generated by AI during triage.*

The AFK agent needs human input before proceeding. See #${hitl_issue_num} for details.

Removing \`${LABEL_READY}\` until resolved." 2>/dev/null || true

  gh issue edit "$issue_num" --repo "$REPO" \
    --remove-label "$LABEL_READY" \
    --remove-label "$LABEL_IN_PROGRESS" \
    --add-label "needs-info" 2>/dev/null || true

  log "Filed HITL issue #$hitl_issue_num for #$issue_num"
  rm -f "$reason_file"
}

# ─── Issue Processing ────────────────────────────────────────────────────────

process_issue() {
  local issue_num="$1"

  # Final gate: refuse to process closed or already-completed issues
  local issue_state
  issue_state=$(gh issue view "$issue_num" --repo "$REPO" --json state -q .state 2>/dev/null || echo "UNKNOWN")
  if [[ "$issue_state" == "CLOSED" ]]; then
    err "Issue #$issue_num is CLOSED — refusing to process"
    mark_completed "$issue_num"
    return 1
  elif [[ "$issue_state" != "OPEN" ]]; then
    err "Issue #$issue_num state unknown ($issue_state) — skipping (transient?)"
    return 1
  fi

  local body
  body=$(gh issue view "$issue_num" --repo "$REPO" --json body -q .body)
  if [[ -z "$body" ]]; then
    err "Could not fetch issue #$issue_num (network/rate-limit?)"
    return 1
  fi
  local target_branch
  target_branch=$(get_target_branch "$body")

  log "═══ Issue #$issue_num → target: $target_branch ═══"

  # Track for cleanup on interrupt
  CURRENT_ISSUE="$issue_num"

  # Claim the issue — add in-progress and remove ready-for-agent atomically
  gh issue edit "$issue_num" --repo "$REPO" \
    --add-label "$LABEL_IN_PROGRESS" \
    --remove-label "$LABEL_READY" 2>/dev/null || true

  # Run the code→review loop
  local loop_exit=0
  run_code_review_loop "$issue_num" "$target_branch" || loop_exit=$?

  case "$loop_exit" in
    0)
      # Approved — commit and land
      local land_exit=0
      commit_and_land "$issue_num" "$target_branch" || land_exit=$?
      if [[ "$land_exit" -ne 0 ]]; then
        err "Landing failed for #$issue_num — keeping work branch ralph/issue-${issue_num}"
        gh issue edit "$issue_num" --repo "$REPO" \
          --remove-label "$LABEL_IN_PROGRESS" \
          --add-label "needs-info" 2>/dev/null || true
        gh issue comment "$issue_num" --repo "$REPO" --body \
          "> *This was generated by AI during triage.*

Code was approved but landing failed (push/merge conflict). Work branch \`ralph/issue-${issue_num}\` preserved for manual resolution." 2>/dev/null || true
        log "✗ Issue #$issue_num — approved but landing failed"
      else
        gh issue edit "$issue_num" --repo "$REPO" \
          --remove-label "$LABEL_IN_PROGRESS" 2>/dev/null || true
        gh issue close "$issue_num" --repo "$REPO" \
          --comment "> *This was generated by AI during triage.*

Implemented and merged via PR by ralph-loop agent.
Code: ${CODING_MODEL} | Review: ${REVIEW_MODEL_1} + ${REVIEW_MODEL_2}" 2>/dev/null || true
        mark_completed "$issue_num"
        log "✓ Issue #$issue_num — completed and landed on $target_branch"
      fi
      ;;
    2)
      # HITL needed — reset tree before filing
      reset_tree
      file_hitl_issue "$issue_num"
      git checkout "$target_branch" 2>/dev/null || git checkout main 2>/dev/null || true
      git branch -D "ralph/issue-${issue_num}" 2>/dev/null || true
      ;;
    *)
      # Failure — reset tree
      reset_tree
      gh issue edit "$issue_num" --repo "$REPO" \
        --remove-label "$LABEL_IN_PROGRESS" \
        --add-label "needs-info" 2>/dev/null || true
      gh issue comment "$issue_num" --repo "$REPO" --body \
        "> *This was generated by AI during triage.*

The AFK agent failed to complete this issue after ${MAX_TOTAL_CYCLES} cycles. Manual review needed." 2>/dev/null || true
      git checkout "$target_branch" 2>/dev/null || git checkout main 2>/dev/null || true
      git branch -D "ralph/issue-${issue_num}" 2>/dev/null || true
      log "✗ Issue #$issue_num — failed"
      ;;
  esac

  # Issue fully processed — clear interrupt tracking
  CURRENT_ISSUE=""
}

# ─── Main Loop ───────────────────────────────────────────────────────────────

run_once() {
  local target_issue="${1:-}"

  local issue_num
  if [[ -n "$target_issue" ]]; then
    issue_num="$target_issue"
  else
    issue_num=$(pick_next_issue)
  fi

  if [[ -z "$issue_num" ]]; then
    log "No ready issues to pick up"
    return 1
  fi

  process_issue "$issue_num"
}

main() {
  local mode="loop"
  local target_issue=""

  while [[ $# -gt 0 ]]; do
    case "$1" in
      --once)      mode="once"; shift ;;
      --dry-run)   mode="dry-run"; shift ;;
      --issue)     target_issue="$2"; mode="once"; shift 2 ;;
      --help|-h)
        cat <<EOF
Usage: $0 [--once | --dry-run | --issue NUM]

AFK dual-agent loop: code (${CODING_MODEL}) → review (${REVIEW_MODEL_1} + ${REVIEW_MODEL_2})
Max ${MAX_REVIEW_CYCLES} cycles per issue. Commits directly to target branch on approval.

Options:
  --once       Process one issue then exit
  --dry-run    Show what would be picked up without acting
  --issue NUM  Work on a specific issue number

Environment:
  RALPH_STATE_DIR        State directory (default: .ralph)
  RALPH_COOLDOWN         Seconds between iterations (default: 30)
  RALPH_MAX_ITERATIONS   Max loops before exit (default: 50)
  RALPH_COPILOT_CMD      Agent command (default: copilot)
EOF
        exit 0
        ;;
      *) err "Unknown option: $1"; exit 1 ;;
    esac
  done

  ensure_deps
  ensure_clean_tree
  init_state
  acquire_lock

  case "$mode" in
    dry-run)
      log "=== Dry Run ==="
      log "Config: code=${CODING_MODEL} review=${REVIEW_MODEL_1}+${REVIEW_MODEL_2} cycles=${MAX_REVIEW_CYCLES}"
      local issues
      issues=$(fetch_ready_issues)
      local count
      count=$(echo "$issues" | jq 'length')
      log "Found $count issues labeled '$LABEL_READY':"
      echo "$issues" | jq -r '.[] | "  #\(.number): \(.title)"'
      log ""
      local next
      next=$(pick_next_issue)
      if [[ -n "$next" ]]; then
        local next_body
        next_body=$(gh issue view "$next" --repo "$REPO" --json body -q .body)
        local next_branch
        next_branch=$(get_target_branch "$next_body")
        log "Would pick: #$next → target branch: $next_branch"
      else
        log "Nothing to pick (all blocked or already processed)"
      fi
      ;;
    once)
      run_once "$target_issue"
      ;;
    loop)
      log "═══ Ralph Loop started (max $MAX_ITERATIONS iterations) ═══"
      log "Config: code=${CODING_MODEL} review=${REVIEW_MODEL_1}+${REVIEW_MODEL_2} cycles=${MAX_REVIEW_CYCLES}"
      log "Repo: $REPO | Label: $LABEL_READY | Cooldown: ${COOLDOWN_SECONDS}s"
      log ""

      read_state
      while [[ "$ITERATION" -lt "$MAX_ITERATIONS" ]]; do
        ITERATION=$((ITERATION + 1))
        update_state "$ITERATION" ""

        if run_once "$target_issue"; then
          log "Cooling down for ${COOLDOWN_SECONDS}s..."
          sleep "$COOLDOWN_SECONDS"
        else
          log "No work available. Waiting 5 minutes..."
          sleep 300
        fi

        read_state
      done

      log "═══ Ralph Loop complete ($ITERATION iterations) ═══"
      ;;
  esac
}

main "$@"
